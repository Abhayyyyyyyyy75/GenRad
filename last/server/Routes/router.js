const express = require("express");
const router = new express.Router();
const controllers = require("../controllers/userControllers");



const formData = require('express-form-data');

// Create an instance of express-form-data
const formMiddleware = formData.parse();

// Use the middleware for parsing form data
router.use(formMiddleware);
// Routes
router.post("/user/register",controllers.userregister);
router.post("/user/sendotp",controllers.userOtpSend);
router.post("/user/login",controllers.userLogin);
router.post("/user/modeloutput",controllers.modeloutput);



module.exports = router;