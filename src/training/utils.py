
class StopTraining:
    def __init__(self, conditions):
        if conditions is None:
            self.operate = None
        else:
            assert "condition" in conditions, "conditions must have value condition"

            condition = conditions.pop("condition")

            assert condition.lower() in [
                "and", "or"], "condition must be either and or or"

            self.operate = all if condition == "and" else any
            self.conditions = conditions
            self.check = True

    def __call__(self, **kwargs):
        if self.operate is None:
            return False

        if self.check:
            assert all(
                cond in kwargs for cond in self.conditions), "Not all selected conditions are available from the training"

        current_state = [
            kwargs[cond] >= value for cond,
            value in self.conditions.items()]

        return self.operate(current_state)
