import { useState } from "react";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [qaPairs, setQaPairs] = useState([
    {
      question:
        "The rise of automation and robotics is another major technological trend that is reshaping industries around the world.",
      options: [
        "Robots will replace all human jobs.",
        "Automation will lead to increased productivity and efficiency.",
        "Robotics and automation will revolutionize manufacturing and agriculture.",
        "The rise of automation and robotics will have no impact on the job market",
      ],
      answer: 1,
    },
  ]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedOption, setSelectedOption] = useState(null);
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleResponseData = (rawData) => {
    try {
      const cleanData = rawData
        .replace(/\\n/g, "")
        .replace(/\\/g, "")
        .replace(/^data:\s*/, "")
        .replace(/^"/, "")
        .replace(/"$/, "");

      console.log("Cleaned Data:", cleanData);

      const parsedObject = JSON.parse(cleanData);

      let qaArray = [];
      if (Array.isArray(parsedObject)) {
        qaArray = parsedObject;
      } else {
        qaArray = [parsedObject];
      }

      console.log("Final QA Array:", qaArray);
      return qaArray;
    } catch (error) {
      console.error("Error parsing response data:", error);
      return [];
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert("Please select a file before submitting.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      setError(null);
      setQaPairs([]);

      const res = await fetch("http://127.0.0.1:5000/generate-qa", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        setLoading(false);
        setError(errorData.error || "An unknown error occurred.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let result = "";

      const stream = new ReadableStream({
        start(controller) {
          function push() {
            reader
              .read()
              .then(({ done, value }) => {
                if (done) {
                  controller.close();
                  // Set loading to false only after the stream is fully processed
                  setLoading(false);
                  return;
                }

                result += decoder.decode(value, { stream: true });

                const dataRegex = /^data: (.*)/;
                const matches = result.match(dataRegex);

                if (matches && matches[1]) {
                  let rawData = matches[1];

                  console.log("Raw Data before cleaning:", rawData);

                  const qaArray = handleResponseData(rawData);

                  if (qaArray.length === 0) {
                    setError("Failed to parse JSON response.");
                    setLoading(false); // Ensure this only happens when there's an error
                    return;
                  }

                  setQaPairs((prev) => [...prev, ...qaArray]);
                  result = ""; // Reset the result buffer after processing data
                }

                push(); // Keep reading until stream is done
              })
              .catch((err) => {
                console.error("Error reading the stream:", err);
                setError(err.message || "Error during stream processing.");
                setLoading(false);
              });
          }
          push(); // Start the reading process
        },
      });

      await new Response(stream); // Await the completion of the streaming process
    } catch (err) {
      console.error("File upload failed:", err);
      setError(err.message || "An error occurred while processing the file.");
      setLoading(false);
    }
  };

  const renderQA = (qa, index) => {
    if (qa && qa.question) {
      return (
        <div key={index}>
          <h3
            style={{
              backgroundColor: "#f4f4f4",
              padding: "5px",
              borderRadius: "5px",
            }}
          >
            <span>
              <strong>{index + 1}:</strong>
            </span>
            {qa.question}
          </h3>
          <ul>
            {qa.options &&
              qa.options.map((option, i) => (
                <li key={i}>
                  <label>
                    <input
                      type="radio"
                      name={`question-${index}`}
                      value={i}
                      checked={
                        selectedOption &&
                        selectedOption.question === index &&
                        selectedOption.answer === i
                      }
                      onChange={() =>
                        setSelectedOption({
                          question: index,
                          answer: i,
                        })
                      }
                      style={{
                        accentColor: i === qa.answer - 1 ? "green" : "red",
                      }}
                    />
                    {option}
                  </label>
                  {selectedOption &&
                    selectedOption.question === index &&
                    selectedOption.answer === i && (
                      <span
                        style={{
                          color: i === qa.answer - 1 ? "green" : "red",
                          marginLeft: "5px",
                        }}
                      >
                        {i === qa.answer - 1 ? "Correct" : "Incorrect"}
                      </span>
                    )}
                </li>
              ))}
          </ul>
        </div>
      );
    }
    return null;
  };

  return (
    <div>
      <h2>Upload PDF for QA Generation</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button type="submit">Submit</button>
      </form>
    
      {error && <p style={{ color: "red" }}>{error}</p>}
      {qaPairs && qaPairs.length > 0 && (
        <div>{qaPairs.map((qa, index) => renderQA(qa, index))}</div>
      )}
      {loading && <p>Loading...</p>}
    </div>
  );
};

export default FileUpload;
