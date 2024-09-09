from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os

class SlackIntegration:
    def __init__(self):
        self.client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        self.channels = {}
        print("SLACK_BOT_TOKEN is set" if "SLACK_BOT_TOKEN" in os.environ else "SLACK_BOT_TOKEN is not set")
        self.create_channels()

    def create_channels(self):
        print("Starting create_channels method")
        channel_names = ["llm-input", "llm-output"]
        try:
            for channel_name in channel_names:
                channel_id = self.create_channel(channel_name)
                if channel_id:
                    self.channels[channel_name] = channel_id
                else:
                    print(f"Failed to create channel: {channel_name}")
        except Exception as e:
            print(f"Error in create_channels: {e}")
        print("Finished create_channels method")

    def create_channel(self, channel_name):
        try:
            print(f"Attempting to create channel: {channel_name}")
            existing_channels = self.client.conversations_list()
            print(f"Existing channels response: {existing_channels}")
            
            for channel in existing_channels["channels"]:
                if channel["name"] == channel_name:
                    print(f"Channel {channel_name} already exists with ID: {channel['id']}")
                    return channel['id']
            
            print(f"Creating new channel: {channel_name}")
            response = self.client.conversations_create(name=channel_name)
            print(f"Channel creation response: {response}")
            
            channel_id = response["channel"]["id"]
            print(f"Created new channel {channel_name} with ID: {channel_id}")
            return channel_id
        except SlackApiError as e:
            print(f"Error creating channel {channel_name}: {e.response['error']}")
            print(f"Full error response: {e.response}")
            return None

    def post_update(self, channel_type, message):
        channel_name = f"llm-{channel_type}"
        if channel_name not in self.channels:
            print(f"Invalid channel type: {channel_type}")
            return

        channel_id = self.channels[channel_name]
        print(f"Attempting to post message to channel: {channel_name} (ID: {channel_id})")
        try:
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=message
            )
            print(f"Message posted to channel {channel_name}. Response: {response}")
        except SlackApiError as e:
            print(f"Error posting message to channel {channel_name}: {e.response['error']}")
            print(f"Full error response: {e.response}")

    def get_channel_ids(self):
        return self.channels

    def check_bot_scopes(self):
        try:
            auth_test = self.client.auth_test()
            bot_id = auth_test["bot_id"]
            bot_info = self.client.bots_info(bot=bot_id)
            scopes = bot_info["bot"]["scopes"]
            print(f"Bot scopes: {scopes}")
            required_scopes = ["channels:manage", "channels:read", "chat:write", "groups:write", "im:write"]
            missing_scopes = [scope for scope in required_scopes if scope not in scopes]
            if missing_scopes:
                print(f"Missing scopes: {', '.join(missing_scopes)}")
            return scopes, not bool(missing_scopes)
        except SlackApiError as e:
            print(f"Error checking bot scopes: {e.response['error']}")
            return [], False
