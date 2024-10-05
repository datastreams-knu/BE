const express = require('express');
const axios = require('axios');
const session = require('express-session');
require('dotenv').config();  // .env 파일에서 환경 변수 로드

const app = express();
const port = 3000;

// 세션 설정
app.use(session({
  secret: process.env.SECRET_KEY,  // 환경 변수에서 비밀키를 가져옴
  resave: false,
  saveUninitialized: true,
}));

// 1. 카카오 로그인 페이지로 리다이렉트
app.get('/auth/kakao', (req, res) => {
  const kakaoAuthURL = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.KAKAO_REST_API_KEY}&redirect_uri=${process.env.KAKAO_REDIRECT_URI}`;
  res.redirect(kakaoAuthURL);
});

// 2. 카카오 로그인 후, 인증 코드 처리 및 액세스 토큰 요청
app.get('/auth/kakao/callback', async (req, res) => {
  const { code } = req.query;

  try {
    // 3. 인증 코드를 통해 카카오 API 서버에서 액세스 토큰 요청
    const tokenResponse = await axios({
      method: 'POST',
      url: 'https://kauth.kakao.com/oauth/token',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      params: {
        grant_type: 'authorization_code',
        client_id: process.env.KAKAO_REST_API_KEY,
        redirect_uri: process.env.KAKAO_REDIRECT_URI,
        code,
      },
    });

    const accessToken = tokenResponse.data.access_token;

    // 4. 액세스 토큰을 사용해 카카오 API로부터 사용자 정보 요청
    const userResponse = await axios({
      method: 'GET',
      url: 'https://kapi.kakao.com/v2/user/me',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });

    const userInfo = userResponse.data;

    // 5. 사용자 정보 처리 및 세션에 저장
    req.session.kakao = {
      id: userInfo.id,
      nickname: userInfo.kakao_account.profile.nickname,
      email: userInfo.kakao_account.email,
    };

    // 로그인 성공 페이지로 리다이렉트
    res.redirect('/profile');
  } catch (error) {
    console.error(error);
    res.status(500).send('카카오 로그인 실패');
  }
});

// 6. 로그인된 사용자 정보 출력 (세션에 저장된 사용자 정보)
app.get('/profile', (req, res) => {
  if (!req.session.kakao) {
    return res.status(403).send('로그인이 필요합니다.');
  }
  
  const { nickname, email } = req.session.kakao;
  res.send(`안녕하세요, ${nickname}님! 이메일: ${email}`);
});

// 7. 로그아웃 처리
app.get('/logout', (req, res) => {
  req.session.destroy();  // 세션 삭제
  res.send('로그아웃되었습니다.');
});

// 서버 시작
app.listen(port, () => {
  console.log(`서버가 http://localhost:${port}에서 실행 중입니다.`);
});
