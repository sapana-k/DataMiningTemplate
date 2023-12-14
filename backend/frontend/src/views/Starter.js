import SalesChart from "../components/dashboard/SalesChart";
// , Button, Form, FormGroup, Label, Input, FormText, 
import { Row, Col, Table, Card, CardTitle, CardBody} from "reactstrap";
import FileUpload from "./FileUpload";
import HttpRequest from "./HttpRequest";
import SelectDataset from "./SelectDataset";

const Starter = () => {
  
  return (
    <div>
      <h1>Clustering</h1> <br/>
      {/* <HttpRequest></HttpRequest> */}
      <Row>
      {/*table*/}
      <Col>
      <FileUpload></FileUpload>
      </Col>
      <SelectDataset/>
      <Col>
      
      {/* <FormGroup>
        <Label for="exampleText">Enter Link</Label>
        <Input id="exampleText" name="text" type="textarea" />
      </FormGroup> */}
    </Col>
      <br></br>
     <Col lg="12">
        <Card>
          <CardTitle tag="h6" className="border-bottom p-3 mb-0">
            <i className="bi bi-card-text me-2"> </i>
            Table with Border
          </CardTitle>
          <CardBody className="">
            <Table bordered>
              <thead>
                <tr>
                  <th>#</th>
                  <th>First Name</th>
                  <th>Last Name</th>
                  <th>Username</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">1</th>
                  <td>Mark</td>
                  <td>Otto</td>
                  <td>@mdo</td>
                </tr>
                <tr>
                  <th scope="row">2</th>
                  <td>Jacob</td>
                  <td>Thornton</td>
                  <td>@fat</td>
                </tr>
                <tr>
                  <th scope="row">3</th>
                  <td>Larry</td>
                  <td>the Bird</td>
                  <td>@twitter</td>
                </tr>
              </tbody>
            </Table>
          </CardBody>
        </Card>
      </Col>
    </Row>
    
    <Row>
      <Col sm="6" lg="6" xl="7" xxl="8">
        {/* <SalesChart /> */}
      </Col>
    </Row>

    </div>
  );
};

export default Starter;
