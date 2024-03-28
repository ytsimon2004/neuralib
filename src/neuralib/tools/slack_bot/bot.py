import functools
import os
import socket
from datetime import datetime
from pathlib import Path

from neuralib.util.util_verbose import fprint

__all__ = ['slack_bot',
           'send_slack_bot']


def slack_bot(timestamp: bool = True):
    """
    sending error message to slack if any error occurs
    """

    def _decorator(f):
        @functools.wraps(f)
        def _bot(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except BaseException as e:
                text = f'`{socket.gethostname()}` ERROR: {e} in `{Path(f.__code__.co_filename).name}`'

                # find opt as first arg, get detailed information
                if len(args) > 0:
                    opt = args[0]
                    exp_date = getattr(opt, 'EXP_DATE', None)
                    animal = getattr(opt, 'ANIMAL', None)

                    if exp_date is not None and animal is not None:
                        text += f'while running DATA: `{exp_date}_{animal}`'

                    if timestamp:
                        text += f'at `{datetime.today().strftime("%y-%m-%d %H:%M:%S")}`'

                send_slack_bot(text)

                raise e

        return _bot

    return _decorator


def send_slack_bot(text: str):
    env_path = Path(__file__).parent / '.env'
    try:
        import slack
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
        try:
            client = slack.WebClient(token=os.environ['SLACK_TOKEN'])
        except KeyError:
            fprint('To get the slack notification, generate token in local machine first', vtype='warning')
        else:
            client.chat_postMessage(channel='#rscvp', text=text)
    except BaseException as e:  # not care, internet disconnected
        fprint(e, vtype='error')
