// MongoDB 스키마 정의
const mongoose = require('mongoose');
const { Schema } = mongoose;

// QUESTION 스키마
const questionSchema = new Schema({
	text: { type: String },
	Question: { type: String },
	Answer: {
		answer: { type: String, required: true },
		disclaimer: { type: String },
		images: [String],
		references: { type: String }
	},
	QDate: { type: Date, default: Date.now }
});

// CHAT 스키마
const chatSchema = new Schema({
	Cname: { type: String },
	Cdate: { type: Date, default: Date.now },
	Questions: [{ type: Schema.Types.ObjectId, ref: 'Question' }]
});

const userSchema = new mongoose.Schema({
	email: {
		type: String,
		required: true,
		unique: true, // 이메일은 중복되지 않아야 함
		trim: true,   // 앞뒤 공백 제거
		match: /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/ // 이메일 형식 검증
	},
	password: {
		type: String,
		required: true,
		minlength: 8 // 최소 8자 제한
	},
	nickname: {
		type: String,
		required: true,
		trim: true, // 앞뒤 공백 제거
		minlength: 2, // 닉네임 최소 길이
		maxlength: 20 // 닉네임 최대 길이
	},
	joinedAt: {
		type: Date,
		default: Date.now // 기본값으로 현재 시간
	},
	Chats: [{ type: Schema.Types.ObjectId, ref: 'Chat' }],
	num_of_question: { type: int, default: 0 }
});

// 각 스키마에 대한 모델 생성
const User = mongoose.model('User', userSchema);
const Chat = mongoose.model('Chat', chatSchema);
const Question = mongoose.model('Question', questionSchema);

module.exports = { User, Chat, Question };
