import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";
import App from "./App.jsx";
import QuizList from "./QuizList.jsx";
import LoadQa from "./load-qa.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/quizzes" element={<QuizList />} />
        <Route path="/take-quiz" element={<LoadQa />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
