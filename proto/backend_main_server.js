const express = require('express');
const mongoose = require('mongoose');
const { User, Chat, Question } = require('./models'); // 모델 불러오기

const app = express();
app.use(express.json()); // JSON 파싱

// MongoDB 연결
const dbURI = 'mongodb://localhost:27017/chatbotDB'; // 로컬 MongoDB URL (필요에 따라 수정)
mongoose.connect(dbURI, { useNewUrlParser: true, useUnifiedTopology: true })
	.then(() => console.log('success : DB connection'))
	.catch((err) => console.log('fail : DB connection', err));

// 사용자 생성 API 예시
app.post('/users', async (req, res) => {
	try {
		const newUser = new User(req.body);
		const savedUser = await newUser.save();
		res.status(201).json(savedUser);
	} catch (error) {
		res.status(400).json({ error: error.message });
	}
});

/*************
채팅 api
*************/
//질문 보내기
app.post('/api/question/:type', async (req, res) => {
	//type jsonify
	//db에 질문 저장
});
//답변 받기
app.get('/api/chat/answer', async (req, res) => {
	//db에 있는 질문 꺼내오기 -> 인자는 뭐로??
	//질문 ai서버에 보낸 후 return
});
//질문 보낸 시간 받기
app.get('api/chat/date', async (req, res) => {
	//db에 있는 질문들 어떻게 구분???
});
//답변 평가하기
app.post('api/chat/reputate', async (req, res) => {

});

/*************
히스토리 api
*************/
//새 히스토리 만들기
app.post('api/history/make', async (req, res) => {

});
//히스토리 이름 수정하기
app.patch('api/history/name/:type', async (req, res) => {

});
//히스토리 내의 모든 채팅받기
app.get('api/history/all', async (req, res) => {

});
//히스토리 시간 받기
app.get('api/history/date', async (req, res) => {

});
//히스토리 삭제하기
app.delete('api/history/date/:name', async (req, res) => {

});


// 서버 실행
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
	console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
});
