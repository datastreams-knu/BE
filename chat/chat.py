from flask import Flask, jsonify, request

app = Flask(__name__)

# 채팅 목록을 반환
@app.route('/api/chats', methods=['GET'])
def get_chats():
    # DB에서 채팅 목록을 불러옴
    chats = Chat.query.all()
    return jsonify([chat.to_dict() for chat in chats])

# 특정 채팅의 메시지를 반환
@app.route('/api/chats/<int:chat_id>/messages', methods=['GET'])
def get_messages(chat_id):
    messages = Message.query.filter_by(chat_id=chat_id).all()
    return jsonify([message.to_dict() for message in messages])

# 메시지 추가
@app.route('/api/chats/<int:chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    data = request.json
    new_message = Message(chat_id=chat_id, user_id=data['user_id'], content=data['content'])
    db.session.add(new_message)
    db.session.commit()
    return jsonify(new_message.to_dict()), 201