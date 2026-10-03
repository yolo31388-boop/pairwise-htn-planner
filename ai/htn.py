"""HTN层次任务网络规划器 - 存在bug"""
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


@dataclass
class PrimitiveAction:
    name: str
    params: dict = field(default_factory=dict)
    effects: Callable[[dict], dict] = None


class HTNPlanner:
    def __init__(self):
        self.methods: dict[str, list[Method]] = {}
        self.actions: dict[str, PrimitiveAction] = {}
        self.world_state: dict = {}

    def register_method(self, method: Method):
        if method.task not in self.methods:
            self.methods[method.task] = []
        self.methods[method.task].append(method)

    def register_action(self, action: PrimitiveAction):
        self.actions[action.name] = action

    def decompose_task(self, task: Task, world_state: dict) -> Optional[list[Task]]:
        """分解复合任务 - BUG: 不检查前置条件"""
        if task.name not in self.methods:
            return None
        # BUG: 不检查前置条件，直接选第一个方法
        method = self.methods[task.name][0]
        return method.subtasks

    def plan(self, task: Task, world_state: dict) -> Optional[list[PrimitiveAction]]:
        """规划 - BUG: 只做简单递归，不做回溯和循环检测"""
        result = []
        self._plan_recursive(task, world_state, result)
        return result if result else None

    def _plan_recursive(self, task: Task, world_state: dict, result: list):
        """递归规划 - BUG: 不做回溯"""
        if task.is_primitive:
            if task.name in self.actions:
                result.append(self.actions[task.name])
            return
        subtasks = self.decompose_task(task, world_state)
        if subtasks is None:
            return
        for subtask in subtasks:
            self._plan_recursive(subtask, world_state, result)

    def execute_method(self, action: PrimitiveAction, world_state: dict) -> dict:
        """执行方法 - BUG: 不检查世界状态变化"""
        if action.effects:
            return action.effects(world_state)
        return world_state

    def validate_network(self) -> bool:
        """验证任务网络 - BUG: 不检查循环引用"""
        visited = set()
        for task_name in self.methods:
            if not self._validate_task(task_name, visited):
                return False
        return True

    def _validate_task(self, task_name: str, visited: set) -> bool:
        # BUG: 不检查循环引用
        if task_name not in self.methods:
            return task_name in self.actions or True
        for method in self.methods[task_name]:
            for subtask in method.subtasks:
                if not subtask.is_primitive:
                    self._validate_task(subtask.name, visited)
        return True
