import React, { useState, useEffect } from 'react';
import axios from 'axios';

function Calculation () {
  const [statistics, setStatistics] = useState({ mean: null});
  const handleCalculation = (data) => {
    setStatistics(data);
    console.log(statistics.mean);
  }
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.post("http://localhost:8000/calculate1/", {
      dataset: name_dataset
    })
    .then((response) => {
      console.log(response);
    });
  })

  return (
    <div>
      <h1>Mean from backend</h1>
      <p>{statistics.mean}</p>
    </div>
  );
}


export default HttpRequest;