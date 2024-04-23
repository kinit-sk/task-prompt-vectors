import torch
from typing import Self


class TaskPrompt:
    task_name: str = None
    task_weights: torch.Tensor = (None,)
    origin_weigts: torch.Tensor = (None,)
    prompt: torch.Tensor = None
    tasks: set = None
    device: str = None

    def __init__(
        self,
        task_name: str,
        task_weights: torch.Tensor = None,
        origin_weigts: torch.Tensor = None,
        prompt: torch.Tensor = None,
        device: str = "cuda",
    ):
        if "+" not in task_name and "-" not in task_name:
            self.task_name = f"+ {task_name}"
        else:
            self.task_name = task_name

        if isinstance(prompt, torch.Tensor):
            self.prompt = prompt
        else:
            assert isinstance(task_weights, torch.Tensor) and isinstance(
                origin_weigts, torch.Tensor
            )

            self.prompt = task_weights.to(device) - origin_weigts.to(device)

        self.tasks = set(task_name.replace("+ ", "").replace("- ", "").split(" "))

    def __add__(self, other: Self) -> Self:
        # print(type(other))
        assert isinstance(other, self.__class__)

        new_prompt = self.prompt + other.prompt
        new_task_name = f"{self.task_name} {other.task_name}"

        return TaskPrompt(new_task_name, prompt=new_prompt)

    def __radd__(self, other: Self) -> Self:
        if other is None or isinstance(other, int):
            return self

        return self.__add__(other)

    def __sub__(self, other: Self) -> Self:
        assert isinstance(other, self.__class__)
        return self + -other

    def __neg__(self):
        if self.task_name[0] == "-":
            new_task_name = f"+{self.task_name[1:]}"
        else:
            new_task_name = f"-{self.task_name[1:]}"

        return TaskPrompt(new_task_name, prompt=-self.prompt)

    def __str__(self):
        return f"{self.task_name} {self.prompt}"

    def apply(self, origin_weights, coef=1):
        return torch.nn.Parameter(origin_weights + coef * self.prompt)
