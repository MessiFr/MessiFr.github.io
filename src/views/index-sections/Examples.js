import React from "react";
import { Link } from "react-router-dom";
// reactstrap components
import { Button, Container, Row } from "reactstrap";

// core components

function Examples() {
  return (
    <>
      <div className="section section-images" data-background-color="black" >
        {/* <div className="space-50"></div> */}
        <Container className="text-center">
          <Row>
            <div className="col">
              
                <img
                  alt="..."
                  className="img-raised"
                  src={require("assets/img/doc-page.jpeg")}
                  to='/documents'
                  width="600px" height="200px"
                ></img>
              
              <Button
                className="btn-round"
                color="default"
                to="/documents"
                outline
                tag={Link}
              >
                View Documents
              </Button>
            </div>
            <div className="col">
              
                <img
                  alt="..."
                  className="img-raised"
                  src={require("assets/img/prof-page.jpeg")}
                  to='/profile-page'
                  width="600px" height="200px"
                ></img>
                
              <Button
                className="btn-round"
                color="default"
                to="/profile-page"
                outline
                tag={Link}
              >
                View Profile
              </Button>
            </div>
            <div className="col">
              
                <img
                  alt="..."
                  className="img-raised"
                  src={require("assets/img/gallery-page.jpeg")}
                  to='/gallery'
                  width="600px" height="200px"
                ></img>
                
              <Button
                className="btn-round"
                color="default"
                to="/gallery"
                outline
                tag={Link}
              >
                View Profile
              </Button>
            </div>
          </Row>
        </Container>
      </div>
    </>
  );
}

export default Examples;
