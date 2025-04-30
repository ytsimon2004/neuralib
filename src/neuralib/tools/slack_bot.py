import os

from slack import WebClient
from slack.errors import SlackApiError

from neuralib.typing import PathLike
from neuralib.util.verbose import fprint

__all__ = ['send_slack_message']


def send_slack_message(env_file: PathLike,
                       message: str) -> None:
    """
    Send message to slack channel

    :param env_file: env file with fields ``SLACK_TOKEN`` (i.e., SLACK_TOKEN=xoxb-<USER_SLACK_TOKEN>),
        and ``CHANNEL_ID`` (i.e., ``#general`` or ``ID``)
    :param message: message to send
    :return:
    """

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)

        try:
            client = WebClient(token=os.environ['SLACK_TOKEN'])
        except KeyError:
            fprint('To get the slack notification, generate token in local machine first', vtype='warning')
        else:
            try:
                client.chat_postMessage(
                    channel=os.environ['CHANNEL_ID'],
                    text=message,
                )
            except SlackApiError as e:
                print(f"Error sending message: {e.response['error']}")

    except BaseException as e:  # not care, internet disconnected
        fprint(e, vtype='error')
