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

// JWT 인증 미들웨어
const authenticateToken = (req, res, next) => {
	// Authorization 헤더에서 토큰 추출
	const authHeader = req.headers['authorization'];
	const token = authHeader && authHeader.split(' ')[1]; // Bearer <token> 형식

	if (!token) {
		return res.status(401).json({ error: 'Access token is required' });
	}

	// 토큰 검증
	jwt.verify(token, JWT_SECRET, (err, user) => {
		if (err) {
			return res.status(403).json({ error: 'Invalid or expired token' });
		}

		// 사용자 정보를 요청 객체에 추가
		req.user = user; // user는 JWT의 payload 정보
		next(); // 다음 미들웨어 또는 라우터로 이동
	});
};

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
		const { email } = req.query; // 쿼리 파라미터에서 email 추출

		if (!email) {
			return res.status(400).json({ error: 'Email is required' });
		}

		const emailRegex = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
		if (!emailRegex.test(email)) {
			return res.status(400).json({ error: 'Invalid email format' });
		}

		// MongoDB에서 이메일 존재 여부 확인
		const existingUser = await User.findOne({ email }).exec();
		const isEmailTaken = !!existingUser; // 이메일이 존재하면 true 반환

		// 결과 반환
		return res.status(200).json({ email_check: isEmailTaken });
	} catch (error) {
		console.error('Error checking email:', error);
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

		// 필수 필드 확인
		if (!email || !password) {
			return res.status(400).json({ error: 'Missing required fields' });
		}

		// 이메일로 사용자 검색
		const user = await User.findOne({ email });
		if (!user) {
			return res.status(401).json({ error: 'Invalid email or password' });
		}

		// 비밀번호 검증
		const isPasswordValid = await bcrypt.compare(password, user.password);
		if (!isPasswordValid) {
			return res.status(401).json({ error: 'Invalid email or password' });
		}

		// JWT 토큰 생성 (payload에 userId와 email 포함)
		const token = jwt.sign(
			{
				userId: user._id,
				email: user.email
			},
			JWT_SECRET,
			{ expiresIn: '24h' }
		);

		// 응답 반환 (토큰 포함)
		return res.status(200).json({
			token
		});
	} catch (error) {
		console.error('Error during login:', error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

// 내 정보 보기
// 닉네임 가입일자 이메일 질문개수
app.get('/api/member/info', authenticateToken, async (req, res) => {
	try {
		const { userId } = req.user;

		if (!userId) {
			return res.status(400).json({ error: 'User ID is required' });
		}

		const user = await User.findById(userId);
		if (!user) {
			return res.status(404).json({ error: 'User not found' });
		}

		return res.status(200).json({
			nickname: user.nickname,
			joinedAt: user.joinedAt.toISOString(),
			num_of_question: user.num_of_question
		});
	} catch (error) {
		console.error(error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

// 회원탈퇴 API
app.delete('/api/member/delete', authenticateToken, async (req, res) => {
	try {
		const { userId } = req.user;

		const user = await User.findById(userId).populate('Chats');
		if (!user) {
			return res.status(404).json({ error: 'User not found' });
		}

		const chatIds = user.Chats.map(chat => chat._id);
		const questionIds = user.Chats.flatMap(chat => chat.Questions);
		await Question.deleteMany({ _id: { $in: questionIds } });

		await Chat.deleteMany({ _id: { $in: chatIds } });

		await user.deleteOne();

		return res.status(200).json({ message: 'User, chats, and related questions deleted successfully.' });
	} catch (error) {
		console.error('Error deleting user:', error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

/*************
채팅 api
*************/
// 비회원 전용 질문 (히스토리 추가 x)
app.post('/api/front-ai-response', async (req, res) => {
	try {
		const { question } = req.body;

		if (!question) {
			return res.status(400).json({ error: 'No question provided' });
		}

		// AI 서버에 질문을 전달하고 응답을 받음
		const api_end_point = ai_server_url + '/ai/ai-response';
		const aiResponse = await axios.post(api_end_point, { question }, { timeout: 30000 });
		console.log(aiResponse.data)
		// 응답 반환
		res.status(200).json(aiResponse.data);
	} catch (error) {
		console.error('Error calling AI server:', error);
		const errorMessage = error.response ? error.response.data : error.message;
		res.status(500).json({ error: errorMessage });
	}

});

//회원 전용 질문 (히스토리 추가)
app.post('/api/chat/user-question/:historyId', authenticateToken, async (req, res) => {
	try {
		const { historyId } = req.params;
		const { question } = req.body;
		const { userId } = req.user;
		if (!userId) {
			return res.status(400).json({ error: 'there is no user ID' });
		}
		if (!question) {
			return res.status(400).json({ error: 'There is no question' });
		}
		if (!historyId) {
			return res.status(400).json({ error: 'There is no history ID' });
		}

		// AI 서버에 질문을 전달하고 응답을 받음
		const api_end_point = ai_server_url + '/ai/ai-response';
		const aiResponse = await axios.post(api_end_point, { question }, { timeout: 30000 });

		// 히스토리에 저장 코드
		const new_question = new Question({
			Question: question,
			Answer: aiResponse.data
		});
		await new_question.save();

		const history = await Chat.findById(historyId);
		if (!history) {
			return res.status(404).json({ error: 'History not found' });
		}

		await Chat.findByIdAndUpdate(historyId, { $push: { Questions: new_question._id } });

		await history.save();

		const user = await User.findById(userId);
		await User.findByIdAndUpdate(userId, { $inc: { num_of_questions: 1 } });
		await user.save();

		// 응답 반환	
		res.status(200).json(aiResponse.data);
	} catch (error) {
		console.error('Error calling AI server:', error);
		const errorMessage = error.response ? error.response.data : error.message;
		res.status(500).json({ error: errorMessage });
	}
});

/*************
히스토리 api
*************/
app.post('/api/history/new-history', authenticateToken, async (req, res) => {
	try {
		const { userId } = req.user;
		if (!userId) {
			return res.status(400).json({ error: 'User ID is required' });
		}

		// 사용자 검색
		const user = await User.findById(userId);
		if (!user) {
			return res.status(404).json({ error: 'User not found' });
		}

		// 새 히스토리 생성 및 저장
		const newChat = new Chat({
			Cname: new Date(Date.now()).toISOString().split('T')[0],
			Cdate: Date.now(),
			Questions: []
		});
		await newChat.save();

		// 사용자 Chats 업데이트 및 저장
		user.Chats.push(newChat._id);
		await user.save();

		// 성공 응답
		res.status(201).json({ new_history_id: newChat._id });
	} catch (err) {
		console.error('Error creating new history:', err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});


// 유저의 모든 히스토리의 기본정보 가져오기
// 반환은 id, name, date
app.get('/api/history/show-all', authenticateToken, async (req, res) => {
	try {
		// authenticateToken에서 토큰 검증 후 사용자 정보 추출
		const { userId } = req.user;


		if (!userId) {
			return res.status(400).json({ error: 'User ID is required' });
		}

		// 사용자 ID로 검색
		const user = await User.findById(userId).populate('Chats');
		if (!user) {
			return res.status(401).json({ error: 'User not found' });
		}

		// Chats의 기본 정보만 반환
		const histories = user.Chats.map(chat => ({
			id: chat._id,
			name: chat.Cname,
			date: chat.Cdate
		}));

		return res.status(200).json(histories);
	} catch (error) {
		console.error('Error fetching user histories:', error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

//히스토리 내부 모든 채팅 불러오기
app.get('/api/history/show-questions/:historyId', authenticateToken, async (req, res) => {
	try {
		const { historyId } = req.params; // URL 파라미터에서 historyId 추출
		if (!historyId) {
			return res.status(400).json({ error: 'There is no history ID' });
		}

		const chat = await Chat.findById(historyId).populate({
			path: 'Questions',
			options: { sort: { QDate: 1 } }
		});
		if (!chat) {
			return res.status(404).json({ error: 'Chat not found' });
		}

		return res.status(200).json(chat.Questions);
	} catch (error) {
		console.error('Error fetching questions from history:', error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});

//히스토리 이름 수정
app.post('/api/history/rename/:historyId/:historyName', authenticateToken, async (req, res) => {
	try {
		const { userId } = req.user;
		const { historyId, historyName } = req.params;
		if (!userId || !historyId || !historyName) {
			return res.status(400).json({ error: 'there is no userId or historyId or historyName' });
		}

		const result = await Chat.updateOne(
			{ _id: historyId },
			{ $set: { Cname: historyName } }
		);

		if (result.matchedCount === 0) {
			return res.status(404).json({ error: 'History not found' });
		}

		if (result.modifiedCount === 0) {
			return res.status(400).json({ error: 'History name is already the same' });
		}

		res.status(200).json({ message: 'History name updated successfully.' });
	} catch (err) {
		console.error(err);
		res.status(500).json({ message: 'Internal backend server error.' });
	}
});

//히스토리 삭제
app.delete('/api/history/delete/:historyId', authenticateToken, async (req, res) => {
	try {
		const { historyId } = req.params; // URL 파라미터에서 historyId 추출
		if (!historyId) {
			return res.status(400).json({ error: 'There is no history ID' });
		}

		// 사용자 검색
		const { userId } = req.user;
		const user = await User.findById(userId).populate({
			path: "Chats",
			populate: { path: "Questions", select: "_id" }
		});
		if (!user) {
			return res.status(404).json({ error: 'User not found' });
		}

		// 사용자의 히스토리 중 이름이 일치하는 항목 검색
		const chatToDelete = user.Chats.find(chat => chat._id.toString() === historyId);
		if (!chatToDelete) {
			return res.status(404).json({ error: 'History not found for this user' });
		}

		// 히스토리가 참조하는 Questions 삭제
		await Question.deleteMany({ _id: { $in: chatToDelete.Questions } });

		// 히스토리 삭제
		await Chat.findByIdAndDelete(chatToDelete._id);

		// 사용자의 Chats 배열에서 해당 히스토리 참조 제거
		user.Chats = user.Chats.filter(chat => chat._id.toString() !== chatToDelete._id.toString());
		await user.save();

		res.status(200).json({ message: 'History deleted successfully.' });
	} catch (error) {
		console.error('Error deleting history:', error);
		return res.status(500).json({ error: 'Internal Server Error' });
	}
});


// 서버 실행
app.listen(PORT, '0.0.0.0', () => {
	console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
});
