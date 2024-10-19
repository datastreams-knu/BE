//app 변수 객체 만드는 로직, 만들어진 app객체에 기능을 하나씩 연결함. 뭐로? app.set으로
var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

//미들 웨어를 연결하는 부분 
//-> 미들웨어란? Express에서 말하는 **미들웨어(Middleware)**는 요청(Request)과 응답(Response) 사이에 위치하여 특정 작업을 수행하는 함수입니다. 
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

// app객체를 모듈로 만드는 코드 = bin/www에서 사용되는 app모듈
module.exports = app;
