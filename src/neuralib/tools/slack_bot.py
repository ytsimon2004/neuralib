"""
Module provide basic function for sending Slack messages with Python
=====================================================================

.. seealso:: `<https://www.datacamp.com/tutorial/how-to-send-slack-messages-with-python>`_

1. Configure a Slack App
-------------------------

    1.1. Create an app from https://api.slack.com/app

    1.2. Find the "OAuth & Permissions" tab to open access (i.e., `chat:write`, `chat:write.customize`, `files:read`, `files:write`)

    1.3. Click "Install to Workspace"

    1.4. See token, which starts with ``xoxb-``

    1.5. In Slack, find the ``Apps`` tab on the left side to add the created app

2. Configure an env file
-------------------------

    2.1. Put it in the repo as a (*.env file), **DO NOT** control it with GIT (add to .gitignore)

    2.2. The env file should contain two keys: ``SLACK_TOKEN`` (xoxb-*) and ``CHANNEL_ID`` to send the message

3. Call the function
---------------------

.. code-block:: python

    from neuralib.tools.slack_bot import send_slack_message
    env_file = ...  # env file path
    send_slack_message(env_file, 'Hello, slack!')



"""
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
