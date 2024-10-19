//routes folder
//주소별 라우터가 저장되어 있음.
//서버의 로직이 들어감. -> end point파일 저장 폴더

//index.js : 라우팅 관리
var express = require('express');
var router = express.Router();
const cors = require('cors');
app.use(cors());

/* GET home page. */
router.get('/', function (req, res, next) {
	res.render('index', { title: 'Express' });
});

module.exports = router;
