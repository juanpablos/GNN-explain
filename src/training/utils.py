
class StopTraining:
    def __init__(self, conditions):
        if conditions is None:
            self.operation = None
        else:
            assert "operation" in conditions, "conditions must have value 'operation'"

            assert conditions["operation"] in [
                "and", "or"], "operation must be either and or or"

            self.operation = all if conditions["operation"] == "and" else any
            self.conditions = conditions["conditions"]
            self.check = True
            self.stop = False
            self.stay = conditions.get("stay", 1)
            self.original_stay = conditions.get("stay", 1)

    def __call__(self, **kwargs):
        if self.operation is None:
            return False

        if self.check:
            assert all(
                cond in kwargs for cond in self.conditions), "Not all selected conditions are available from the training"

        current_state = [
            kwargs[cond] >= value for cond,
            value in self.conditions.items()]

        if self.operation(current_state):
            self.stay -= 1
        else:
            self.stay = self.original_stay

        return self.stay < 0
