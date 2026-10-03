"""HTN层次任务网络规划器

修复内容:
- 任务分解: 检查前置条件、按优先级排序、部分分解失败时回溯
- 规划器: 启发式搜索、循环检测、规划缓存
- 方法执行: 检查世界状态一致性、支持中断、失败回退
- 网络验证: 循环引用检测、参数匹配检查、非法状态修改检查
- 复合任务: 支持顺序/并行/选择/循环分解
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Task:
    name: str
    params: dict = field(default_factory=dict)
    is_primitive: bool = False


@dataclass
class Method:
    name: str
    task: str
    preconditions: list[Callable[[dict], bool]] = field(default_factory=list)
    subtasks: list[Task] = field(default_factory=list)
    priority: float = 0.0
    params: Optional[list[str]] = None
    allowed_effects: Optional[set] = None
    mode: str = "sequence"  # sequence | parallel | choice | loop
    repeat: Optional[int] = None
    until: Optional[Callable[[dict], bool]] = None


@dataclass
class PrimitiveAction:
    name: str
    params: dict = field(default_factory=dict)
    effects: Optional[Callable[[dict], dict]] = None
    preconditions: list[Callable[[dict], bool]] = field(default_factory=list)
    modifies: Optional[set] = None
    fallback: Optional["PrimitiveAction"] = None


class HTNPlanner:
    def __init__(self, interrupt_check: Optional[Callable[[], bool]] = None,
                 max_loop_iterations: int = 100):
        self.methods: dict[str, list[Method]] = {}
        self.actions: dict[str, PrimitiveAction] = {}
        self.world_state: dict = {}
        self.interrupt_check = interrupt_check
        self.interrupt_requested = False
        self.max_loop_iterations = max_loop_iterations
        self._plan_cache: dict = {}

    def register_method(self, method: Method):
        if method.task not in self.methods:
            self.methods[method.task] = []
        self.methods[method.task].append(method)
        self.clear_cache()

    def register_action(self, action: PrimitiveAction):
        self.actions[action.name] = action
        self.clear_cache()

    # ---------- 中断控制 ----------

    def request_interrupt(self):
        """请求中断执行（例如被攻击时）"""
        self.interrupt_requested = True

    def reset_interrupt(self):
        self.interrupt_requested = False

    def _interrupted(self) -> bool:
        if self.interrupt_requested:
            return True
        return self.interrupt_check is not None and self.interrupt_check()

    # ---------- 缓存 ----------

    def clear_cache(self):
        self._plan_cache.clear()

    @staticmethod
    def _state_key(world_state: dict):
        def freeze(value):
            if isinstance(value, dict):
                return tuple(sorted((k, freeze(v)) for k, v in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(v) for v in value)
            if isinstance(value, set):
                return tuple(sorted(freeze(v) for v in value))
            try:
                hash(value)
            except TypeError:
                return repr(value)
            return value

        try:
            return tuple(sorted((k, freeze(v)) for k, v in world_state.items()))
        except TypeError:
            return repr(world_state)

    # ---------- 前置条件与启发式排序 ----------

    @staticmethod
    def _preconditions_hold(preconditions, world_state: dict) -> bool:
        return all(pre(world_state) for pre in preconditions)

    def _ordered_methods(self, task_name: str, world_state: dict) -> list[Method]:
        """启发式排序: 优先级高者优先，其次子任务少（代价低）者优先"""
        indexed = list(enumerate(self.methods.get(task_name, [])))
        indexed.sort(key=lambda pair: (-pair[1].priority, len(pair[1].subtasks), pair[0]))
        return [method for _, method in indexed]

    # ---------- 任务分解 ----------

    def decompose_task(self, task: Task, world_state: dict) -> Optional[list[Task]]:
        """分解复合任务：检查前置条件，按启发式排序选择方法"""
        if task.name not in self.methods:
            return None
        for method in self._ordered_methods(task.name, world_state):
            if self._preconditions_hold(method.preconditions, world_state):
                return list(method.subtasks)
        return None

    def decompose_compound(self, task: Task, world_state: dict,
                           mode: Optional[str] = None) -> Optional[list[PrimitiveAction]]:
        """分解复合任务，支持 sequence/parallel/choice/loop，失败时回溯"""
        if task.is_primitive:
            action = self.actions.get(task.name)
            return [action] if action is not None else None
        if task.name not in self.methods:
            return None
        for method in self._ordered_methods(task.name, world_state):
            if not self._preconditions_hold(method.preconditions, world_state):
                continue
            plan = self._expand_method(method, world_state, (task.name,), mode=mode)
            if plan is not None:
                return plan
        return None

    # ---------- 规划 ----------

    def plan(self, task: Task, world_state: dict) -> Optional[list[PrimitiveAction]]:
        """启发式搜索 + 循环检测 + 规划缓存"""
        cache_key = (task.name, self._state_key(world_state))
        if cache_key in self._plan_cache:
            cached = self._plan_cache[cache_key]
            return list(cached) if cached is not None else None
        result = self._search(task, world_state, ())
        self._plan_cache[cache_key] = tuple(result) if result is not None else None
        return result

    def _search(self, task: Task, world_state: dict, path: tuple) -> Optional[list[PrimitiveAction]]:
        if task.is_primitive:
            action = self.actions.get(task.name)
            if action is None:
                return None
            if not self._preconditions_hold(action.preconditions, world_state):
                return None
            return [action]
        if task.name in path:
            # 循环检测：当前分解路径上已出现过该任务
            return None
        if task.name not in self.methods:
            return None
        for method in self._ordered_methods(task.name, world_state):
            if not self._preconditions_hold(method.preconditions, world_state):
                continue
            plan = self._expand_method(method, world_state, path + (task.name,))
            if plan is not None:
                return plan
        return None

    def _expand_method(self, method: Method, world_state: dict, path: tuple,
                       mode: Optional[str] = None) -> Optional[list[PrimitiveAction]]:
        mode = mode or method.mode
        if mode == "sequence":
            return self._expand_sequence(method.subtasks, world_state, path)
        if mode == "parallel":
            return self._expand_parallel(method.subtasks, world_state, path)
        if mode == "choice":
            return self._expand_choice(method.subtasks, world_state, path)
        if mode == "loop":
            return self._expand_loop(method, world_state, path)
        return None

    def _expand_sequence(self, subtasks, world_state, path):
        """顺序分解：任一子任务失败则整体失败，由调用方回溯尝试其他方法"""
        plan = []
        for subtask in subtasks:
            sub_plan = self._search(subtask, world_state, path)
            if sub_plan is None:
                return None
            plan.extend(sub_plan)
        return plan

    def _expand_parallel(self, subtasks, world_state, path):
        """并行分解：独立子任务分别规划，结果交错合并"""
        plans = []
        for subtask in subtasks:
            sub_plan = self._search(subtask, world_state, path)
            if sub_plan is None:
                return None
            plans.append(sub_plan)
        merged = []
        index = 0
        while True:
            progressed = False
            for sub_plan in plans:
                if index < len(sub_plan):
                    merged.append(sub_plan[index])
                    progressed = True
            if not progressed:
                break
            index += 1
        return merged

    def _expand_choice(self, subtasks, world_state, path):
        """选择分解：依次尝试各分支，返回第一个可成功的分支"""
        for subtask in subtasks:
            sub_plan = self._search(subtask, world_state, path)
            if sub_plan is not None:
                return sub_plan
        return None

    def _expand_loop(self, method: Method, world_state, path):
        """循环分解：按 repeat 次数或 until 条件重复展开，迭代而非递归"""
        plan = []
        state = dict(world_state)
        iterations = 0
        while iterations < self.max_loop_iterations:
            if method.until is not None and method.until(state):
                break
            if method.repeat is not None and iterations >= method.repeat:
                break
            if method.until is None and method.repeat is None and iterations >= 1:
                break
            iteration_plan = []
            for subtask in method.subtasks:
                sub_plan = self._search(subtask, state, path)
                if sub_plan is None:
                    return plan if plan else None
                iteration_plan.extend(sub_plan)
            plan.extend(iteration_plan)
            for action in iteration_plan:
                if action.effects is not None:
                    state = action.effects(state)
            iterations += 1
        return plan

    # ---------- 方法执行 ----------

    def execute_method(self, action: PrimitiveAction, world_state: dict) -> dict:
        """执行动作：检查中断、前置条件与世界状态一致性，失败时保持原状态"""
        _, new_state = self._try_execute(action, world_state)
        return new_state

    def execute_plan(self, actions: list[PrimitiveAction], world_state: dict) -> dict:
        """按计划顺序执行：支持中断，动作失败时尝试备用方案"""
        state = dict(world_state)
        for action in actions:
            if self._interrupted():
                break
            success, new_state = self._try_execute(action, state)
            if not success and action.fallback is not None:
                success, new_state = self._try_execute(action.fallback, state)
            if not success:
                break
            state = new_state
        return state

    def _try_execute(self, action: PrimitiveAction, world_state: dict):
        if self._interrupted():
            return False, world_state
        if not self._preconditions_hold(action.preconditions, world_state):
            return False, world_state
        if action.effects is None:
            return True, world_state
        new_state = action.effects(dict(world_state))
        if not self._state_consistent(world_state, new_state):
            return False, world_state
        return True, new_state

    @staticmethod
    def _state_consistent(old_state: dict, new_state: dict) -> bool:
        """世界状态一致性：非负数值资源不允许被修改为负数"""
        for key, new_value in new_state.items():
            old_value = old_state.get(key)
            if (isinstance(old_value, (int, float)) and not isinstance(old_value, bool)
                    and isinstance(new_value, (int, float)) and not isinstance(new_value, bool)):
                if old_value >= 0 and new_value < 0:
                    return False
        return True

    # ---------- 网络验证 ----------

    def validate_network(self) -> bool:
        """验证任务网络：引用存在、无循环引用、参数匹配、状态修改合法"""
        for methods in self.methods.values():
            for method in methods:
                for subtask in method.subtasks:
                    if subtask.is_primitive:
                        if subtask.name not in self.actions:
                            return False
                    elif subtask.name not in self.methods:
                        return False

        # 循环引用检测（三色DFS）
        color = {name: 0 for name in self.methods}

        def has_cycle(name: str) -> bool:
            color[name] = 1
            for method in self.methods.get(name, []):
                for subtask in method.subtasks:
                    if subtask.is_primitive or subtask.name not in self.methods:
                        continue
                    if color[subtask.name] == 1:
                        return True
                    if color[subtask.name] == 0 and has_cycle(subtask.name):
                        return True
            color[name] = 2
            return False

        for name in self.methods:
            if color[name] == 0 and has_cycle(name):
                return False

        # 参数匹配：子任务传入的参数必须在方法声明的参数范围内
        for methods in self.methods.values():
            for method in methods:
                if method.params is None:
                    continue
                declared = set(method.params)
                for subtask in method.subtasks:
                    if not set(subtask.params) <= declared:
                        return False

        # 状态修改检查：子树中动作声明的修改不能超出方法允许的范围
        for task_name, methods in self.methods.items():
            for method in methods:
                if method.allowed_effects is None:
                    continue
                allowed = set(method.allowed_effects)
                for action in self._reachable_actions(task_name):
                    if action.modifies and not set(action.modifies) <= allowed:
                        return False
        return True

    def _reachable_actions(self, task_name: str) -> list[PrimitiveAction]:
        found = []
        visited = set()
        stack = [task_name]
        while stack:
            name = stack.pop()
            if name in visited:
                continue
            visited.add(name)
            for method in self.methods.get(name, []):
                for subtask in method.subtasks:
                    if subtask.is_primitive:
                        action = self.actions.get(subtask.name)
                        if action is not None:
                            found.append(action)
                    else:
                        stack.append(subtask.name)
        return found
