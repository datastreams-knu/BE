// MongoDB 스키마 정의
const mongoose = require('mongoose');
const { Schema } = mongoose;

// USER 스키마
const userSchema = new Schema({
	UId: { type: String, required: true, unique: true },
	Token: { type: String, required: true },
	chats: [chatSchema]
});

// CHAT 스키마
const chatSchema = new Schema({
	CId: { type: String, required: true, unique: true },
	Cdate: { type: Date, default: Date.now },
	question: [questionSchema]
});

// QUESTION 스키마
const questionSchema = new Schema({
	QId: { type: String, required: true, unique: true },
	Question: { type: String, required: true },
	Answer: { type: String, required: true },
	Qdate: { type: Date, default: Date.now }
});

// 각 스키마에 대한 모델 생성
const User = mongoose.model('User', userSchema);
const Chat = mongoose.model('Chat', chatSchema);
const Question = mongoose.model('Question', questionSchema);

module.exports = { User, Chat, Question };
