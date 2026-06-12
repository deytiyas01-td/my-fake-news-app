const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');

const app = express();
const port = 3000;

// Configure file upload storage in memory
const upload = multer({ storage: multer.memoryStorage() });

// Serve static frontend files (HTML, CSS, JS) out of the public folder
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Route to handle analysis submissions from your UI dashboard
app.post('/analyze', upload.single('image'), async (req, res) => {
    try {
        const headline = req.body.headline || "";
        const form = new FormData();
        form.append('headline_text', headline);

        if (req.file) {
            form.append('image_file', req.file.buffer, {
                filename: req.file.originalname,
                contentType: req.file.mimetype,
            });
        }

        // Forward the data directly to your running Python FastAPI server
        const response = await axios.post('http://127.0.0.1:8000/predict', form, {
            headers: { ...form.getHeaders() }
        });

        res.json(response.data);
    } catch (error) {
        console.error("Error communicating with AI Engine:", error.message);
        res.status(500).json({ error: "Could not connect to the Python AI engine backend." });
    }
});

app.listen(port, () => {
    console.log(`\n🌐 Web interface running seamlessly at http://localhost:3000`);
});