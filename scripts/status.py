"""
A status check for all python files.

>>> python status.py

"""


from pathlib import Path

_DIR = '.'


def file_status(file: str) -> str:
    with open(file, 'r') as f:
        try:
            status_line = f.readlines()[1]
            if not 'STATUS' in status_line:
                return 'NO STATUS'
            else:
                return status_line.split(' ', 1)[1].replace('\n', '')
        except IndexError:
            return 'NO STATUS'

def status() -> None:
    print('')
    print('{:<70}  {}'.format('FILE', 'STATUS\n'))
    path_list = Path(_DIR).glob('**/*.py')
    for path in path_list:
        file = str(path)
        if file.endswith('.py') and not file.startswith('status'):
            s = file_status(file)
            print('{:<70}  {}'.format(file, s))
    print('\n')


if __name__ == '__main__':
    status()
