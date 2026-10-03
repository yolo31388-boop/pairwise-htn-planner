"""HTN层次任务网络规划器"""
import heapq
import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional


def _freeze(value):
    """将任意值转换为可哈希形式，用于缓存键和状态比较"""
    if isinstance(value, dict):
        return tuple(sorted(((k, _freeze(v)) for k, v in value.items()), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


@dataclass
class Task:
    name: str
    params: dict = field(default_factory=dict)
    is_primitive: bool = False
    # 复合任务结构：sequence(顺序) / parallel(并行) / choice(选择) / loop(循环)
    task_type: str = "sequence"
    subtasks: list["Task"] = field(default_factory=list)
    condition: Optional[Callable[[dict], bool]] = None  # loop 的继续条件
    max_iterations: int = 100  # loop 最大迭代次数（防止无限循环）
    times: int = 1  # loop 固定重复次数（未指定 condition 时生效）


@dataclass
class Method:
    name: str
    task: str
    preconditions: list[Callable[[dict], bool]] = field(default_factory=list)
    subtasks: list[Task] = field(default_factory=list)
    priority: int = 0  # 方法优先级，高的先尝试
    params: Any = None  # 方法要求的参数名（list 或 dict）
    allowed_effects: Optional[list] = None  # 允许修改的世界状态键，None 表示不限制


@dataclass
class PrimitiveAction:
    name: str
    params: dict = field(default_factory=dict)
    effects: Callable[[dict], dict] = None
    cost: float = 1.0
    modifies: list = field(default_factory=list)  # 声明会修改的世界状态键


class HTNPlanner:
    def __init__(self):
        self.methods: dict[str, list[Method]] = {}
        self.actions: dict[str, PrimitiveAction] = {}
        self.world_state: dict = {}
        self.interrupted: bool = False
        self._plan_cache: dict = {}

    def register_method(self, method: Method):
        if method.task not in self.methods:
            self.methods[method.task] = []
        self.methods[method.task].append(method)
        self._plan_cache.clear()

    def register_action(self, action: PrimitiveAction):
        self.actions[action.name] = action
        self._plan_cache.clear()

    def clear_cache(self):
        """清空规划缓存"""
        self._plan_cache.clear()

    def interrupt(self):
        """中断当前执行（例如被攻击时）"""
        self.interrupted = True

    def resume(self):
        """解除中断"""
        self.interrupted = False

    # ---------- 任务分解 ----------

    def decompose_task(self, task: Task, world_state: dict = None) -> Optional[list[Task]]:
        """分解复合任务：检查前置条件，按优先级选择方法"""
        if world_state is None:
            world_state = self.world_state
        if task.name not in self.methods:
            return None
        applicable = self._applicable_methods(task, world_state)
        if not applicable:
            return None
        return list(applicable[0].subtasks)

    def _applicable_methods(self, task: Task, world_state: dict) -> list[Method]:
        """返回前置条件满足的方法，按优先级和效率排序"""
        candidates = []
        for index, method in enumerate(self.methods.get(task.name, [])):
            try:
                satisfied = all(pre(world_state) for pre in method.preconditions)
            except Exception:
                continue  # 前置条件求值异常视为不满足
            if satisfied:
                # 方法排序：优先级高、子任务少（效率高）的先尝试，同级保持定义顺序
                candidates.append((-method.priority, len(method.subtasks), index, method))
        candidates.sort(key=lambda item: item[:3])
        return [item[3] for item in candidates]

    # ---------- 复合任务分解（顺序/并行/选择/循环） ----------

    def decompose_compound(self, task: Task, world_state: dict = None) -> Optional[list[Task]]:
        """分解结构化复合任务，支持并行、选择、循环分解"""
        if world_state is None:
            world_state = self.world_state
        if task.task_type == "choice":
            # 选择分解：尝试所有分支，返回第一个可行的，而不是永远选第一个
            for branch in task.subtasks:
                if self._branch_viable(branch, world_state):
                    return [branch]
            return None
        if task.task_type == "loop":
            # 循环分解：按条件/次数展开子任务
            result = []
            current = task
            while True:
                expansion = self._expand_structured(current, world_state)[0]
                if not expansion:
                    break
                result.extend(expansion[:-1])
                current = expansion[-1]
            return result
        # 顺序/并行分解：并行子任务相互独立，可任意顺序执行，统一返回全部
        return list(task.subtasks)

    def _expand_structured(self, task: Task, state: dict) -> list[list[Task]]:
        """返回结构化任务的所有可能展开方式（供搜索回溯）"""
        if task.task_type == "choice":
            return [[branch] for branch in task.subtasks]
        if task.task_type == "loop":
            if task.condition is not None:
                try:
                    if not task.condition(state):
                        return [[]]
                except Exception:
                    return [[]]
                fuel = task.max_iterations
            else:
                fuel = task.times
            if fuel <= 0:
                return [[]]
            nxt = replace(task, max_iterations=task.max_iterations - 1,
                          times=task.times - 1)
            return [[*task.subtasks, nxt]]
        # sequence / parallel：线性化为同一批子任务
        return [list(task.subtasks)]

    def _branch_viable(self, task: Task, world_state: dict) -> bool:
        """检查分支在当前状态下是否可行"""
        if task.is_primitive:
            return task.name in self.actions
        if task.subtasks or task.task_type != "sequence":
            return bool(task.subtasks)
        return bool(self._applicable_methods(task, world_state))

    # ---------- 规划 ----------

    def plan(self, task: Task, world_state: dict = None) -> Optional[list[PrimitiveAction]]:
        """启发式搜索规划，带循环检测和规划缓存"""
        if world_state is None:
            world_state = self.world_state
        cache_key = (self._task_key(task), _freeze(world_state))
        if cache_key in self._plan_cache:
            cached = self._plan_cache[cache_key]
            return list(cached) if cached else None
        result = self._search(task, world_state)
        self._plan_cache[cache_key] = result
        return list(result) if result else None

    def _search(self, root_task: Task, world_state: dict,
                max_expansions: int = 10000) -> Optional[list[PrimitiveAction]]:
        """A* 启发式搜索：迭代式（不会栈溢出），失败自动回溯，visited 集合检测循环

        max_expansions 限制搜索节点数，保证无限/超大搜索空间也能在有限时间内返回。
        """
        counter = itertools.count()
        heap = [(self._heuristic((root_task,)), 0.0, next(counter),
                 world_state, (root_task,), ())]
        visited = set()
        nodes_expanded = 0
        while heap:
            _, cost, _, state, remaining, actions = heapq.heappop(heap)
            if not remaining:
                return list(actions)
            nodes_expanded += 1
            if nodes_expanded > max_expansions:
                return None  # 超出搜索预算，放弃
            node_key = (_freeze(state), tuple(self._task_key(t) for t in remaining))
            if node_key in visited:
                continue  # 循环检测：相同状态 + 相同剩余任务，无需重复搜索
            visited.add(node_key)
            head, rest = remaining[0], remaining[1:]
            if head.is_primitive:
                action = self.actions.get(head.name)
                if action is None:
                    continue
                new_state = self._apply_effects(action, state)
                if new_state is None:
                    continue
                new_cost = cost + action.cost
                heapq.heappush(heap, (
                    new_cost + self._heuristic(rest), new_cost, next(counter),
                    new_state, rest, actions + (action,),
                ))
                continue
            if head.subtasks or head.task_type != "sequence":
                expansions = self._expand_structured(head, state)
            else:
                # 部分分解回溯：所有前置条件满足的方法都入队，失败分支自动回退
                expansions = [list(m.subtasks)
                              for m in self._applicable_methods(head, state)]
            for subtasks in expansions:
                new_remaining = tuple(subtasks) + rest
                heapq.heappush(heap, (
                    cost + self._heuristic(new_remaining), cost, next(counter),
                    state, new_remaining, actions,
                ))
        return None

    def _heuristic(self, remaining_tasks) -> float:
        """启发函数：估计剩余任务数，引导优先搜索更接近目标的分支"""
        return float(len(remaining_tasks))

    def _apply_effects(self, action: PrimitiveAction, state: dict) -> Optional[dict]:
        """应用动作效果，失败或产生非法状态时返回 None"""
        if action.effects is None:
            return state
        try:
            new_state = action.effects(state)
        except Exception:
            return None
        if new_state is None or not self._state_valid(new_state):
            return None
        return new_state

    @staticmethod
    def _state_valid(state: dict) -> bool:
        """世界状态合法性检查：资源类数值不允许为负"""
        for value in state.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < 0:
                    return False
        return True

    def _task_key(self, task: Task):
        return (
            task.name,
            _freeze(task.params),
            task.is_primitive,
            task.task_type,
            task.max_iterations,
            task.times,
            task.condition,
            tuple(self._task_key(st) for st in task.subtasks),
        )

    # ---------- 方法执行 ----------

    def execute_method(self, action, world_state: dict,
                       interrupt_check: Optional[Callable[[], bool]] = None) -> dict:
        """执行动作：检查世界状态变化，支持中断，失败时保持原状态"""
        # 可中断：被攻击等情况下停止执行
        if self.interrupted or (interrupt_check is not None and interrupt_check()):
            return world_state
        if isinstance(action, str):
            action = self.actions.get(action)
        if action is None:
            return world_state
        # 失败处理：效果执行异常或世界状态被改为非法状态时，回退到原状态
        new_state = self._apply_effects(action, world_state)
        if new_state is None:
            return world_state
        return new_state

    # ---------- 任务网络验证 ----------

    def validate_network(self) -> bool:
        """验证任务网络：循环引用、参数匹配、状态修改检查"""
        done = set()
        for task_name in self.methods:
            if self._has_cycle(task_name, frozenset(), done):
                return False
        for methods in self.methods.values():
            for method in methods:
                for subtask in method.subtasks:
                    if subtask.is_primitive:
                        if not self._check_state_modification(method, subtask):
                            return False
                    else:
                        if subtask.name not in self.methods and not subtask.subtasks:
                            return False  # 引用的复合任务没有注册方法
                        if not self._check_param_match(subtask):
                            return False
        return True

    def _has_cycle(self, task_name: str, path: frozenset, done: set) -> bool:
        """DFS 检测任务引用环（A -> B -> A）"""
        if task_name in path:
            return True
        if task_name in done:
            return False
        if task_name not in self.methods:
            return False
        path = path | {task_name}
        for method in self.methods[task_name]:
            for subtask in method.subtasks:
                if not subtask.is_primitive:
                    if self._has_cycle(subtask.name, path, done):
                        return True
        done.add(task_name)
        return False

    def _check_param_match(self, subtask: Task) -> bool:
        """参数匹配：子任务提供的参数需满足至少一个方法的参数要求"""
        sub_methods = self.methods.get(subtask.name)
        if not sub_methods:
            return True
        provided = set(subtask.params.keys())
        return any(self._required_params(m) <= provided for m in sub_methods)

    @staticmethod
    def _required_params(method: Method) -> set:
        params = method.params
        if not params:
            return set()
        if isinstance(params, dict):
            return set(params.keys())
        return set(params)

    def _check_state_modification(self, method: Method, subtask: Task) -> bool:
        """状态修改检查：动作修改的状态键必须在方法允许的范围内"""
        if method.allowed_effects is None:
            return True
        action = self.actions.get(subtask.name)
        if action is None:
            return True
        return set(action.modifies) <= set(method.allowed_effects)
