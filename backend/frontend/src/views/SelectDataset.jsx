import React, { useState} from 'react'
import {Button, Form, FormGroup, Label, Input } from "reactstrap";
import axios from 'axios';

const SelectDataset = () => {
    const [statistics, setStatistics] = useState({ mean: null});
    
    const handleChange = async(e) => {
        console.log("this is chosen", e.target.value)
        try {
          await axios.post("http://localhost:8000/api/calculate2/",
              {dataset: e.target.value, m: "bhanu"}
          ).then((response) => {
              console.log(response.status, response.data);
              setStatistics(response.data);
          });
      }
      catch (error) {
          console.log(error);
      }
    }

    // const handleCalculation = async() => {
    //     console.log("button pressed, calling")
    // }

  return (
    <div>
    <Form>
      <FormGroup>
      <Label for="exampleSelect">
        Select
      </Label>
      <Input
        id="exampleSelect"
        name="select"
        type="select"
        onChange={handleChange}
      >
      <option>
        Iris Dataset
      </option>
      <option>
        Breast Cancer Dataset
      </option>
      </Input>
      </FormGroup>
      {/* <Button className="mt-2" onSubmit={handleCalculation}>Submit</Button> */}
      <p>{statistics.mean}</p>
    </Form>
    </div>
  )
}

export default SelectDataset