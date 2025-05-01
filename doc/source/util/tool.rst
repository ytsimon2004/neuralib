Utility Tool
==================
Provide utility tool from other API


Google SpreadSheet
---------------------
Wrapper for gspread

.. seealso::

    - `Source Github <https://github.com/burnash/gspread>`_
    - `Source Doc <https://docs.gspread.org/en/latest/>`_


- **Refer to API**: :doc:`../api/neuralib.tools.gspread`



Slack Notification
--------------------
Module provide basic function for sending Slack messages with Python

**Example Use Case**

1. notification when running analysis in server (error encountered ...)

2. notification for animal training process

.. seealso::

    `<https://www.datacamp.com/tutorial/how-to-send-slack-messages-with-python>`_

Configure a Slack App
^^^^^^^^^^^^^^^^^^^^^^^^^^^

    1.1. Create an app from https://api.slack.com/app

    1.2. Find the "OAuth & Permissions" tab to open access (i.e., `chat:write`, `chat:write.customize`, `files:read`, `files:write`)

    1.3. Click "Install to Workspace"

    1.4. See token, which starts with ``xoxb-``

    1.5. In Slack, find the ``Apps`` tab on the left side to add the created app

Configure an env file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

    2.1. Put it in the repo as a (*.env file), **DO NOT** control it with GIT (add to .gitignore)

    2.2. The env file should contain two keys: ``SLACK_TOKEN`` (xoxb-*) and ``CHANNEL_ID`` to send the message

Call the function
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from neuralib.tools.slack_bot import send_slack_message
    env_file = ...  # env file path
    send_slack_message(env_file, 'Hello, slack!')
