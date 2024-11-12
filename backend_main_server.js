//dependencies
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');
const PORT = 3001;

const { User, Chat, Question } = require('./models'); // 모델 불러오기

//express 객체 생성.
const app = express();
app.use(express.json()); // JSON 파싱	

//cors 설정
app.use(cors());

//REST-api url
const ai_server_url = 'http://13.210.175.149:5000';
const kakao_server_url = 'https://kauth.kakao.com';
const kakao_redirect_uri = 'https://www.knu-chatbot.site/callback';

//카카오 로그인용 keys
const kakao_rest_api_key = '4eb17d6035355206cad120867a29beac';

//로그 확인용 미들웨어
app.use((req, res, next) => {
	console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
	console.log('Headers:', req.headers);
	next();
});

// MongoDB 연결
const dbURI = 'mongodb://localhost:27017/chatDB';
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
로그인 api
*************/
app.post('/api/login', async (req, res) => {
	//1. get /oauth/authorize
	//format : '/oauth/authorize?response_type=code&client_id=${REST_API_KEY}&redirect_uri=${REDIRECT_URI}'
	//2. 인가코드 받기, 반드시 email사용 동의 항목 넣어야 함.	
	const authorization_code = await axios.get(kakao_server_url + '/oauth/authorize?response_type=code&client_id=${kakao_rest_api_key}&redirect_uri=${kakao_redirect_uri}');
	//3. post /oauth/token
	//const kakao_token = await axios.post
	//4. token으로 사용자 email가져옴.
	//5. email이 db에 있다면, 로그인 없다면 회원가입시킴.
});

/*************
채팅 api
*************/
// 질문을 받아 AI 서버에 전달하고 응답을 반환하는 API 엔드포인트
app.post('/api/front-ai-response', async (req, res) => {
	try {
		const { question } = req.body;

		if (!question) {
			return res.status(400).json({ error: 'No question provided' });
		}

		// AI 서버에 질문을 전달하고 응답을 받음
		const aiServerUrl = ai_server_url + '/api/ai-response'; // AI 서버의 IP 주소로 변경 필요
		// 받은 응답은 json형식
		const aiResponse = await axios.post(aiServerUrl, { question });
		const { response } = aiResponse.data;

		res.status(200).json(response);
	} catch (error) {
		console.error('Error calling AI server:', error);
		res.status(500).json({ error: error.message });
	}
});

//필요한 것: history id -> body에?
// history id는 어디에 저장되어 있어야 하는가?
/*
request body 예시 (json)
{
	"userID"  : "userid",  -----> user id 대신 kakaotalk email로 인증받자. 액세스토큰 -> api에서 email가져오기
	"historyID" : "historyid",
	"question" : "question contents"
}

result body 예시 (json)
{
	"response" : "responses"
}
*/
app.post('/api/front-ai-response-loggined', async (req, res) => {
	try {
		const { question } = req.body;

		if (!question) {
			return res.status(400).json({ error: 'No question provided' });
		}

		// AI 서버에 질문을 전달하고 응답을 받음
		const aiServerUrl = ai_server_url + '/api/ai-response'; // AI 서버의 IP 주소로 변경 필요
		// 받은 응답은 json형식
		const aiResponse = await axios.post(aiServerUrl, { question });
		const { response } = aiResponse.data;

		res.status(200).json({ response });
		//사용자 토큰이나 id를 받아서 알맞은 history에 저장.

	} catch (error) {
		console.error('Error calling AI server:', error);
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
