// const express = require('express') // common js method  
import express from 'express'		// moudule js method 

const app = express()

app.get('/', function (req, res) {
  res.send('Hello World')
})

app.listen(3000)