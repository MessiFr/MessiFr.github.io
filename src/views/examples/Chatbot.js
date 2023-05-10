import React, { useState } from "react";
// import openai from "openai";
const { Configuration, OpenAIApi } = require("openai");

const configuration = new Configuration({
  // apiKey: process.env.OPENAI_API_KEY,
    apiKey: "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",
});
const openai = new OpenAIApi(configuration);

function Chatbot() {
  const [inputText, setInputText] = useState("");
  const [responseText, setResponseText] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    // const prompt = `User: ${inputText}\nAI:`;
    const response = await openai.createCompletion({
      model: "text-davinci-003",
      // prompt: "\"\"\"\nUtil exposes the following:\nutil.openai() -> authenticates & returns the openai module, which has the following functions:\nopenai.Completion.create(\n    prompt=\"<my prompt>\", # The prompt to start completing from\n    max_tokens=123, # The max number of tokens to generate\n    temperature=1.0 # A measure of randomness\n    echo=True, # Whether to return the prompt in addition to the generated completion\n)\n\"\"\"\nimport util\n\"\"\"\nCreate an OpenAI completion starting from the prompt \"Once upon an AI\", no more than 5 tokens. Does not include the prompt.\n\"\"\"\n",
      prompt: `${inputText}`,
      temperature: 0,
      max_tokens: 64,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      stop: ["\"\"\""],
    });
    setResponseText(response.data.choices[0].text);
    setInputText("");
  };

  return (
    <div>
      <h1>Chatbot</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <button type="submit">Send</button>
      </form>
      <p>{responseText}</p>
    </div>
  );
}

export default Chatbot;
