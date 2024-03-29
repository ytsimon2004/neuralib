For sending message to slack automatically 
---
* refer to https://www.youtube.com/watch?v=KJ5bFv-IRFM

### How to start?
1. generate OAuth Tokens in slack api
2. copy Tokens to `SLACK_TOKEN` in `.env` in the base directory in `slack_bot` module
3. ```bash
    pip install slackclient
    pip install python-dotenv
   ```
4. Add slack `#{channel}` to APP
5. use `slack_bot` decorator