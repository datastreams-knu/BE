//dependencies
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const PORT = 3001;

const { User, Chat, Question } = require('./models'); // 모델 불러오기

//cors 설정 
const corsOptions = {
	origin: 'https://www.knu-chatbot.site',
	methods: ['GET', 'POST', 'OPTIONS'],
	allowedHeaders: ['Content-Type', 'Authorization'],
};

//express 객체 생성.
const app = express();
app.use(express.json()); // JSON 파싱	
app.use(cors(corsOptions));

// MongoDB 연결
const dbURI = 'mongodb://localhost:27017/chatDB'; // 로컬 MongoDB URL (필요에 따라 수정)
mongoose.connect(dbURI)
	.then(() => console.log('success : DB connection'))
	.catch((err) => console.log('fail : DB connection', err));

// 사용자 생성 임시 api
app.post('/api/user/add', async (req, res) => {
	try {
		const newUser = new User({
			"UId": "1111",
			"Token": "22222",
			"Chats": []
		});
		const savedUser = await newUser.save();
		res.status(201).json(savedUser);
	} catch (error) {
		res.status(400).json({ error: error.message });
	}
});

//frontend - backend test api
app.get('/api/test', (req, res) => {
	console.log('GET /api/test 요청 수신');
	res.json({
		success: true,
		message: 'Backend communication successful!',
		timestamp: new Date().toISOString(),
	});
});

/*************
채팅 api
*************/
//질문 보내기
app.post('/api/question', async (req, res) => {
	try {
		const { userId, questionText } = req.body;
		const user = await User.findOne({ UId: userId });
		if (!user) return res.status(404).json({ message: 'User not found.' });

		const newQuestion = new Question({
			QText: questionText,
			QDate: new Date(),
		});
		await newQuestion.save();

		user.Questions.push(newQuestion._id);
		await user.save();

		res.status(201).json(newQuestion);
	} catch (error) {
		res.status(500).json({ error: error.message });
	}
});

//답변 받기
app.get('/api/chat/answer', async (req, res) => {
	try {
		const { questionId } = req.query;
		const question = await Question.findById(questionId);
		if (!question) return res.status(404).json({ message: 'Question not found.' });

		// AI 서버 통신 로직 필요 (예: AI 서버 API 호출 및 응답 수신)
		const aiResponse = "This is a placeholder for the AI response.";

		question.answer = aiResponse;
		await question.save();

		res.status(200).json({ answer: aiResponse });
	} catch (error) {
		res.status(500).json({ error: error.message });
	}
});

//질문 보낸 시간 받기
app.get('/api/chat/date', async (req, res) => {
	try {
		const { questionId } = req.query;
		const question = await Question.findById(questionId);
		if (!question) return res.status(404).json({ message: 'Question not found.' });

		res.status(200).json({ date: question.QDate });
	} catch (error) {
		res.status(500).json({ error: error.message });
	}
});

//답변 평가하기
app.post('/api/chat/reputate/:reputation', async (req, res) => {
	try {
		const { questionId } = req.body;
		const { reputation } = req.params;
		const question = await Question.findById(questionId);
		if (!question) return res.status(404).json({ message: 'Question not found.' });

		question.reputation = reputation;
		await question.save();

		res.status(200).json({ message: 'Reputation updated successfully.' });
	} catch (error) {
		res.status(500).json({ error: error.message });
	}
});

/*************
히스토리 api
*************/
//새 히스토리 만들기
app.post('/api/history/make', async (req, res) => {
	const { userId } = req.body;

	try {
		const user = await User.findOne({ UId: userId });
		if (!user) return res.status(404).json({ message: 'User not found.' });

		const newChat = new Chat({
			CId: mongoose.Types.ObjectId(),
			Cname: 'initial name',
			Cdate: Date.now(),
			Questions: []
		});

		await newChat.save();
		user.Chats.push(newChat._id);
		await user.save();

		res.status(201).json(newChat);
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

//히스토리 이름 수정하기
app.patch('/api/history/name/:after', async (req, res) => {
	const { user_id, chat_id } = req.body;
	const { after } = req.params;

	try {
		const user = await User.findOne({ UId: user_id });
		if (!user) return res.status(404).json({ message: 'User not found.' });

		const chat = await Chat.findOne({ CId: chat_id });
		if (!chat) return res.status(404).json({ message: 'Chat not found.' });

		chat.Cname = after;
		await chat.save();

		res.json({ message: 'Name changed successfully.' });
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

//히스토리 내의 모든 채팅받기
app.get('/api/history/all', async (req, res) => {
	const { user_id, chat_id } = req.query;

	try {
		const user = await User.findOne({ UId: user_id }).populate('Chats');
		if (!user) return res.status(404).json({ message: 'User not found.' });

		const chat = user.Chats.find(c => c.CId === chat_id);
		if (!chat) return res.status(404).json({ message: 'Chat not found.' });

		res.status(200).json(chat);
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

//히스토리 시간 받기
app.get('/api/history/date', async (req, res) => {
	const { user_id, chat_id } = req.query;

	try {
		const user = await User.findOne({ UId: user_id }).populate('Chats');
		if (!user) return res.status(404).json({ message: 'User not found.' });

		const chat = user.Chats.find(c => c.CId === chat_id);
		if (!chat) return res.status(404).json({ message: 'Chat not found.' });

		res.status(200).json({ date: chat.Cdate });
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

//히스토리 삭제하기
app.delete('/api/history/date/:name', async (req, res) => {
	const { user_id, chat_id } = req.body;

	try {
		const user = await User.findOne({ UId: user_id });
		if (!user) return res.status(404).json({ message: 'User not found.' });

		const chatIndex = user.Chats.findIndex(c => c.toString() === chat_id);
		if (chatIndex === -1) return res.status(404).json({ message: 'Chat not found.' });

		user.Chats.splice(chatIndex, 1);
		await user.save();

		await Chat.deleteOne({ CId: chat_id });
		res.status(200).json({ message: 'Chat deleted successfully.' });
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

// 서버 실행
app.listen(PORT, '0.0.0.0', () => {
	console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
});
