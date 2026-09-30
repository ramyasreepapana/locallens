# LocalLens - 2-minute demo script

Setup before recording: run `streamlit run app.py`, open http://localhost:8501, close other tabs. Run `python benchmark.py` once and note the timing from YOUR machine (if you want to quote one).

**0:00 - Problem (20 s)**
"We all photograph documents: invoices, letters, forms. Online OCR tools upload those images to someone else's server. That's a real privacy risk, and it needs internet."

**0:20 - The idea (15 s)**
"LocalLens reads documents with AI entirely on your own PC. Nothing is uploaded. There's no cloud API and no account."

**0:35 - Run it (35 s)**
"I'll click Use sample image." *(click; the spinner appears)* "A neural-network OCR model is running right here through ONNX Runtime. The model files ship inside the install, so there's nothing to download."
*(results appear)* "It found the text regions, drew boxes on the image, and extracted the text. Green boxes are confident; red means double-check."

**1:10 - Fields and honesty (20 s)**
"Below, simple pattern matching pulls out the email, phone number, date, amount and link. And the app is honest about its limits: it flags low-confidence lines, and the model can misread characters, so you always review the result."

**1:30 - Local proof (15 s)**
"The badges and the line under them show what actually happened: processed locally, no cloud API, and which ONNX Runtime providers ran the models." *(Say "CPU" if that's what it shows.)*

**1:45 - Snapdragon (10 s)**
"It's built on ONNX Runtime, which is how Qualcomm's QNN plugin reaches the Snapdragon NPU. We've included an NPU backend that refuses to start unless a real NPU device exists. On our test machine, it ran on the CPU, and we report exactly that."
*(If you verified NPU execution on a Snapdragon laptop, replace this with what you actually measured.)*

**1:55 - Close (5 s)**
"LocalLens: useful AI for sensitive documents, with the documents staying on your device. Thank you."
