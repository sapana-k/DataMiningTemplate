import { Row, Col, Table, Card, CardTitle, CardBody, Button, Form, FormGroup, Label, Input, FormText, } from "reactstrap";
import React, { useState } from "react";

const FileUpload = () => {
    
    const [selectedFile, setSelectedFile] = useState(null);

    const changeHandler = (e) => {
        setSelectedFile(e.target.files[0]);
    };

    const handleSubmission = (e) => {
        e.preventDefault(); // Prevent the default form submission
        if (!selectedFile) {
            // Handle case where no file is selected
            console.log("no file selected")
            return;
        } 
        const file = selectedFile;
        console.log(file)
        const formData = new FormData();
        formData.append('file', file);
        try {
            fetch(
            'http://localhost:8000/api/calculate1/',
            {
                method: 'POST',
                body: formData,
            }
            )
            .then((response) => response.json())
            .then((result) => {
                console.log('Success:', result);
            // onUpload(result);
            })
            .catch((error) => {
            console.error('Sapnaaaaaa Error:', error);
            });
        }
        catch (error) {
            console.error('sapnaaaa Error uploading file:', error);
        }
    };

    return (
    <div>
    <Form encType="multipart/form-data" onSubmit={handleSubmission}>
      <FormGroup>
      <Label for="exampleFile">File</Label>
      <Input id="exampleFile" name="file" type="file"  onChange={changeHandler}/>
      <FormText>
        Enter dataset
      </FormText>
      </FormGroup>
      <Button className="mt-2" type="submit">Submit</Button>
      </Form>

      <Card></Card>
    </div>
    );
}

export default FileUpload;

