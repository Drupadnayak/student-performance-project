# QUICK START GUIDE - 5 Minutes Setup

## ⚡ Get Running in 5 Steps

### Step 1️⃣: Install Python Packages (1 minute)
```bash
cd student-performance-project
pip install -r requirements.txt
```

### Step 2️⃣: Train the Model (1 minute)
```bash
python train_model.py
```
✅ You should see: "✅ MODEL TRAINING COMPLETE!"

### Step 3️⃣: Start Flask Server (30 seconds)
```bash
python app.py
```
✅ You should see: "🚀 Starting Flask server..."

### Step 4️⃣: Open in Browser (10 seconds)
Navigate to:
```
http://localhost:5000
```

### Step 5️⃣: Make a Prediction (30 seconds)
- Enter Study Hours: **8**
- Enter Attendance: **85**
- Enter Previous Score: **75**
- Click: **"🚀 Predict Score"**

✅ **DONE!** You should see the prediction result!

---

## 🎯 Expected Output

```
PREDICTED FINAL SCORE: 78.45
STATUS: ✅ PASS
CONFIDENCE: 78.45%
```

---

## 📁 Files Created

✅ `train_model.py` - Model training script
✅ `app.py` - Flask backend
✅ `templates/index.html` - Frontend form
✅ `static/style.css` - Styling
✅ `requirements.txt` - Dependencies
✅ `README.md` - Full documentation
✅ `models/model.pkl` - (Generated after training)
✅ `models/scaler.pkl` - (Generated after training)

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run `pip install -r requirements.txt` |
| "Model not found" | Run `python train_model.py` first |
| "Port 5000 in use" | Change port in `app.py` to 5001 |
| "Permission denied" | Use `python3` instead of `python` |

---

## 🔗 Key Files to Show Examiner

1. **train_model.py** - Complete ML pipeline
2. **app.py** - Flask backend with predictions
3. **templates/index.html** - User-friendly interface
4. **models/model.pkl** - Trained model (binary file)
5. **README.md** - Full documentation

---

## 📊 Viva Key Points

**"In simple terms, my system works like this:**
1. I collected student data (hours studied, attendance, previous scores)
2. I trained a Linear Regression model to find the pattern
3. I built a web form where students enter their details
4. The model instantly predicts their final score
5. If score ≥ 40, it's PASS, else FAIL"

---

## ✨ You're All Set! 🎉

Your Student Performance Prediction System is complete and ready for submission!

**Files checked:** ✅
**Model trained:** ✅
**Flask app running:** ✅
**Frontend working:** ✅
**Documentation complete:** ✅

---

**Total time to complete setup: ~5 minutes**
**Ready for viva: YES ✅**
