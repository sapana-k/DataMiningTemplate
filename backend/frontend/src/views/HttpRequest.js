import React, { useState, useEffect } from 'react';
import axios from 'axios';

function HttpRequest() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.get('<http://localhost:8000/calculate1/>')
      .then(response => {
        setMessage(response.data.msg);
      })
      .catch(error => {
        console.log(error);
      });
  }, []);

  return (
    <div>
      <h1>Hello, World!</h1>
      <p>{message}</p>
    </div>
  );
}


export default HttpRequest;