//dependecies
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const PORT = 3001;

const { User, Chat, Question } = require('./models'); // 모델 불러오기

//express 객체 생성.
const app = express();
app.use(express.json()); // JSON 파싱	
app.use(cors()); // 모든 도메인에서 접근 허용

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
	//db에 질문 저장
});
//답변 받기
app.get('/api/chat/answer', async (req, res) => {
	//db에 있는 질문 꺼내오기 -> 인자는 뭐로??
	//질문 ai서버에 보낸 후 return
});
//질문 보낸 시간 받기
app.get('/api/chat/date', async (req, res) => {
	//db에 있는 질문들 어떻게 구분???
});
//답변 평가하기
app.post('/api/chat/reputate/:reputation', async (req, res) => {

});

/*************
히스토리 api
*************/
//새 히스토리 만들기
app.post('api/history/make', async (req, res) => {
	//처음 이름은 어떻게 하지? 새로운 ChatID는 어떻게 정의????
	// 프론트앤드에서 CId는 어떻게 저장하지?
	const { userId } = req.body;

	try {
		const user = await User.findOne({ UId: userId });
		if (!user) {
			return res.status(404).json({ message: 'User not found.' });
		}

		// 새로운 Chat 객체 생성
		const newChat = new Chat({
			CId: CId,
			Cname: 'initial name',
			Cdate: Date.now(),
			Question: []
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
// after : 수정할 히스토리 이름.
// body에는 수정할 히스토리의 id, 유저의 id가 필요함.
app.patch('/api/history/name/:after', async (req, res) => {
	const { user_id, chat_id } = req.body;
	const { after } = req.params;

	try {
		const user = await User.findOne({ UId: user_id });
		if (!user)
			return res.status(404).json({ message: 'User not found.' });
		const chat = await Chat.findOne({ CId: chat_id });

		if (!chat)
			return res.status(404).json({ message: 'Chat not found.' });

		// db접근 후 히스토리 이름 변경 및 저장.
		chat.Cname = after;
		await chat.save();

		//성공 시 응답.
		res.json({ message: 'name has been changed successly.' });
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });

	}

});

//히스토리 내의 모든 채팅받기
app.get('/api/history/all', async (req, res) => {
	const { user_id, chat_id } = req.body;

	try {
		// 유효한 user 인지 판단.
		const user = await User.findOne({ UId: user_id });
		if (!user)
			return res.status(404).json({ message: 'User not found.' });

		const chats = await user.populate('Chats');
		if (!chats)
			return res.status(404).json({ message: 'Chat not found.' });

		res.status(200).json(chats);
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

//히스토리 시간 받기
app.get('/api/history/date', async (req, res) => {
	const { user_id, chat_id } = req.body;

	try {
		// 유효한 user 인지 판단.
		const user = await User.findOne({ UId: user_id });
		if (!user)
			return res.status(404).json({ message: 'User not found.' });

		const chats = user.populate('Chats')
		if (!chats)
			return res.status(404).json({ message: 'Chat not found.' });

		res.status(200).chats;
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});
//히스토리 삭제하기
app.delete('/api/history/date/:name', (req, res) => {


});


// 서버 실행
app.listen(PORT, '0.0.0.0', () => {
	console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
});
