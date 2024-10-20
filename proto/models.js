// MongoDB 스키마 정의
const mongoose = require('mongoose');
const { Schema } = mongoose;

// QUESTION 스키마
const questionSchema = new Schema({
	Question: { type: String, required: true },
	Answer: { type: String, required: true },
	Qdate: { type: Date, default: Date.now }
});

// CHAT 스키마
const chatSchema = new Schema({
	Cname: { type: String },
	Cdate: { type: Date, default: Date.now },
	Questions: [{ type: Schema.Types.ObjectId, ref: 'Question' }]
});

// USER 스키마
const userSchema = new Schema({
	Token: { type: String, required: true },
	Chats: [{ type: Schema.Types.ObjectId, ref: 'Chat' }]
});


// 각 스키마에 대한 모델 생성
const User = mongoose.model('User', userSchema);
const Chat = mongoose.model('Chat', chatSchema);
const Question = mongoose.model('Question', questionSchema);

module.exports = { User, Chat, Question };
