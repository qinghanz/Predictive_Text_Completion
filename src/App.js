import React, { useState, useEffect } from "react";

function App() {
  const [text, setText] = useState("");
  const [predictedText, setPredictedText] = useState("");
  const [backendStatus, setBackendStatus] = useState("Connecting...");

  useEffect(() => {
    // Check backend connection status
    fetch("/")
      .then(() => setBackendStatus("Connected"))
      .catch(() => setBackendStatus("Disconnected"));
  }, []);

  const handleInputChange = (event) => {
    const newText = event.target.value;
    setText(newText);

    // Send text to the backend for prediction
    fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: newText }),
    })
      .then((response) => response.json())
      .then((data) => {
        const predictedText = data.predictedText;
        setPredictedText(predictedText);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  return (
    <div className="container">
      <h1 className="title">Predictive Text App</h1>
      <div className="status">
        <strong>Backend Status:</strong> {backendStatus}
      </div>
      <textarea
        className="input"
        value={text}
        onChange={handleInputChange}
        rows="5"
        cols="50"
      ></textarea>
      <div className="predicted-text">
        <h2>Predicted Text:</h2>
        <p>{predictedText}</p>
      </div>
      <style jsx>{`
        .container {
          max-width: 600px;
          margin: 0 auto;
          padding: 20px;
          text-align: center;
        }

        .title {
          font-size: 24px;
          margin-bottom: 20px;
        }

        .status {
          font-size: 18px;
          margin-bottom: 20px;
        }

        .input {
          width: 100%;
          padding: 10px;
          margin-bottom: 20px;
        }

        .predicted-text {
          font-size: 20px;
        }
      `}</style>
    </div>
  );
}

export default App;
