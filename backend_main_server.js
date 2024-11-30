//dependencies
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const PORT = 3001;

const { User, Chat, Question } = require('./models'); // 모델 불러오기

//express 객체 생성.
const app = express();
app.use(express.json()); // JSON 파싱	

//cors 설정
app.use(cors());

//ai 서버 url
const ai_server_url = 'http://127.0.0.1:5000';

//jwt key
const JWT_SECRET = 'dev_jwt_secret_key_G7v^#JdLwP!8qFzT@X9p$';

//로그 확인용 미들웨어
app.use((req, res, next) => {
	console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
	console.log('Headers:', req.headers);
	next();
});

// MongoDB 연결
const dbURI = 'mongodb://localhost:27017/chatDB'; // 로컬 MongoDB URL (필요에 따라 수정)
mongoose.connect(dbURI)
	.then(() => console.log('success : DB connection'))
	.catch((err) => console.log('fail : DB connection', err));

//frontend - backend test api
app.get('/api/test', (req, res) => {
	console.log('GET /api/test 요청 수신');
	res.json({
		success: true,
		message: 'Front - Backend communication successful!',
		timestamp: new Date().toISOString(),
	});
});

/*************
로그인, 회원가입 api
*************/

// 이메일 중복 확인
app.get('/api/member/check-email', async (req, res) => {
	try {
		const { email } = req.query.email;

		const emailRegex = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
		if (!email || !emailRegex.test(email)) {
			return res.status(400).json({ error: 'Invalid email format' });
		}

		const existingUser = await User.findOne({ email });
		const email_check = !!existingUser; // 이메일이 존재하면 true 반환

		return res.status(200).json({ email_check: email_check });
	} catch (error) {
		console.error(error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

// 회원가입
app.post('/api/member/signup', async (req, res) => {
	try {
		const { email, password, nickname } = req.body;

		if (!email || !password || !nickname) {
			return res.status(400).json({ error: 'Missing required fields' });
		}

		// 이메일 중복 확인 생략 (프론트엔드에서 이미 중복 확인 버튼을 통해 검증했음을 가정)
		//const existingUser = await User.findOne({ email });
		//if (existingUser) {
		//	return res.status(409).json({ error: 'Email already exists' });
		//}

		// 비밀번호 해싱
		const hashedPassword = await bcrypt.hash(password, 10);

		// 새 사용자 생성
		const newUser = new User({
			email,
			password: hashedPassword,
			nickname,
			joinedAt: new Date()
		});

		await newUser.save();
		return res.status(201).send(); // 성공 상태 코드 반환
	} catch (error) {
		console.error(error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

// 로그인
app.post('/api/member/login', async (req, res) => {
	try {
		const { email, password } = req.body;

		if (!email || !password) {
			return res.status(400).json({ error: 'Missing email or password' });
		}

		// 사용자 찾기
		const user = await User.findOne({ email });
		if (!user) {
			return res.status(401).json({ error: 'Invalid email or password' });
		}

		// 비밀번호 확인
		const isPasswordValid = await bcrypt.compare(password, user.password);
		if (!isPasswordValid) {
			return res.status(401).json({ error: 'Invalid email or password' });
		}

		// JWT 토큰 생성
		const accessToken = jwt.sign({ userId: user._id }, JWT_SECRET, { expiresIn: '24h' });

		return res.status(200).json({ accessToken });
	} catch (error) {
		console.error(error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

// 내 정보 보기
app.get('/api/member/info', async (req, res) => {
	try {
		const { userId } = req.query;

		if (!userId) {
			return res.status(400).json({ error: 'User ID is required' });
		}

		const user = await User.findById(userId);
		if (!user) {
			return res.status(404).json({ error: 'User not found' });
		}

		return res.status(200).json({
			nickname: user.nickname,
			joinedAt: user.joinedAt.toISOString()
		});
	} catch (error) {
		console.error(error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

/*************
채팅 api
*************/
// 비회원 전용 질문 (히스토리 추가 x)
app.post('/api/front-ai-response', async (req, res) => {
	try {
		console.log('Request body:', req.body);  // 추가 로그
		const { question } = req.body;

		if (!question) {
			return res.status(400).json({ error: 'No question provided' });
		}

		// AI 서버에 질문을 전달하고 응답을 받음
		const api_end_point = ai_server_url + '/ai/ai-response';
		const aiResponse = await axios.post(api_end_point, { question }, { timeout: 30000 });

		// 응답 반환
		res.status(200).json(aiResponse);
	} catch (error) {
		console.error('Error calling AI server:', error);
		const errorMessage = error.response ? error.response.data : error.message;
		res.status(500).json({ error: errorMessage });
	}

});

//회원 전용 질문 (히스토리 추가)
app.post('/api/chat/user-question', async (req, res) => {
	try {
		console.log('Request body:', req.body);  // 추가 로그
		const { question, token } = req.body;
		const decoded = jwt.verify(token, JWT_SECRET);

		if (!decoded) {
			return res.status(401).json({ error: 'Unauthorized' });
		}

		if (!question) {
			return res.status(400).json({ error: 'No question provided' });
		}

		// AI 서버에 질문을 전달하고 응답을 받음
		const api_end_point = ai_server_url + '/ai/ai-response';
		const aiResponse = await axios.post(api_end_point, { question }, { timeout: 30000 });

		// 히스토리에 저장 코드
		// code here

		// 응답 반환
		res.status(200).json(aiResponse);
	} catch (error) {
		console.error('Error calling AI server:', error);
		const errorMessage = error.response ? error.response.data : error.message;
		res.status(500).json({ error: errorMessage });
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
	const { email } = req.query;

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
