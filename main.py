from flask import Flask, render_template, request, jsonify
from model_handler import ModelHandler
from slack_integration import SlackIntegration
import os

app = Flask(__name__)

model_handler = ModelHandler()
slack_integration = SlackIntegration()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.json['input_text']
    output = model_handler.generate_response(input_text)
    
    # Post updates to Slack
    slack_integration.post_update("input", input_text)
    slack_integration.post_update("output", output)
    
    return jsonify({'response': output})

@app.route('/model_structure')
def model_structure():
    structure = model_handler.get_model_structure()
    return jsonify(structure)

@app.route('/model_stats')
def model_stats():
    stats = model_handler.get_model_stats()
    return jsonify(stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = 'large_text_file.txt'
        file.save(filename)
        
        # Load the uploaded file
        model_handler.load_file(filename)
        
        # Process the uploaded file
        result = model_handler.process_large_file()
        
        return jsonify({'message': result})

@app.route('/process_large_file', methods=['POST'])
def process_large_file():
    result = model_handler.process_large_file()
    return jsonify({'message': result})

@app.route('/test_optimized_model', methods=['POST'])
def test_optimized_model():
    input_text = request.json['input_text']
    output = model_handler.generate_response(input_text)
    return jsonify({'response': output})

@app.route('/test_slack')
def test_slack():
    channel_ids = slack_integration.get_channel_ids()
    slack_integration.post_update("input", "Test input message")
    slack_integration.post_update("output", "Test output message")
    return f"Test messages sent to Slack. Channel IDs: {channel_ids}"

@app.route('/debug_slack')
def debug_slack():
    channel_ids = slack_integration.get_channel_ids()
    slack_token_set = "SLACK_BOT_TOKEN" in os.environ
    scopes, has_required_scopes = slack_integration.check_bot_scopes()
    
    debug_info = {
        "channel_ids": channel_ids,
        "slack_token_set": slack_token_set,
        "num_channels_created": len(channel_ids),
        "expected_channels": ["llm-input", "llm-output"],
        "missing_channels": [ch for ch in ["llm-input", "llm-output"] if ch not in channel_ids],
        "has_required_scopes": has_required_scopes,
        "actual_scopes": scopes
    }
    
    return jsonify(debug_info)

@app.route('/test_slack_detailed')
def test_slack_detailed():
    try:
        # Check bot scopes
        scopes, has_required_scopes = slack_integration.check_bot_scopes()
        
        if not has_required_scopes:
            return jsonify({
                "success": False,
                "error": "Missing required Slack bot scopes",
                "message": "Please update the Slack app permissions and reinstall the app",
                "actual_scopes": scopes
            })
        
        # Create channels
        slack_integration.create_channels()
        
        # Get channel IDs
        channel_ids = slack_integration.get_channel_ids()
        
        # Post test messages
        slack_integration.post_update("input", "Detailed test input message")
        slack_integration.post_update("output", "Detailed test output message")
        
        return jsonify({
            "success": True,
            "has_required_scopes": has_required_scopes,
            "actual_scopes": scopes,
            "channel_ids": channel_ids,
            "message": "Detailed Slack test completed successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Detailed Slack test failed"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
