import os


class Listen:
    def __init__(self, filename='command.txt') -> None:
        self.commands = {'lr': None, 'save': False, 'exit': False}
        self.filename = filename

    def exists(self):
        return True if self.filename is not None and os.path.isfile(
            self.filename) else False

    def read(self):
        if self.exists():
            with open(self.filename, 'r') as f:
                command = f.readline().rstrip()
            os.remove(self.filename)
            if (command.find('lr') != -1):
                # change learning rate
                split = command.split("=")
                if len(split) == 2:
                    try:
                        lr = float(split[1])
                    except Exception:
                        print(f'Can\'t understand lr command: "{split[1]}"')
                        return False
                    self.commands['lr'] = lr
                    return True
                else:
                    return False
            elif (command.find('save') != -1):
                self.commands['save'] = True
                return True
            elif (command.find('exit') != -1):
                self.commands['save'] = True
                self.commands['exit'] = True
                return True
            else:
                return False
        else:
            return False

    def check(self, var=None):
        self.read()
        if var is not None:
            if var == 'lr' or var == 'LR' or var == 'Learning rate':
                output = self.commands['lr']
                self.commands['lr'] = None
                return output
            elif var == 'save':
                output = self.commands['save']
                self.commands['save'] = False
                return output
            elif var == 'exit':
                output = self.commands['exit']
                self.commands['exit'] = False
                return output
        return False
