class StepDownScheduler():
    def __init__(
        self,
        optimizer,
        initial_epoch=0,
        config={
            'multiplier': {
                'value': 0.1
            },
            'triggers': {
                'value': [20, 50, 100]
            }
        }
    ) -> None:
        self.multiplier = config['multiplier']['value']
        self.optimizer = optimizer
        self.triggers = config['triggers']['value']
        self.counter = 0
        for self.counter in range(initial_epoch):
            self.step()

    def step(self):
        if self.counter in self.triggers:
            lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr * self.multiplier
        self.counter += 1
