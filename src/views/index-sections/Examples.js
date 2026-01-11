// import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./example.css"
import { useState, useEffect } from "react";

// reactstrap components
import { Button, Container } from "reactstrap";
import { Grid } from "@mui/material";


function Examples() {

  const [width, setWidth] = useState(600);
  const [largeScreen, setLargeScreen] = useState(true);

  useEffect(() => {
    if (window.innerWidth < 1000) {
      setWidth(600);
      setLargeScreen(false);
    }
  }, [])

  return (
    <>
    
      <div className="section section-images" data-background-color="black" >
        {/* <div className="space-50"></div> */}
        <Container className="text-center" style={{marginBottom:"100vh"}}>
          <Grid container spacing={2} justify="center" alignItems="center">
            {largeScreen ? (
              <Grid item xs={12} md={4}>
              
              <img
                alt="..."
                className="img-raised"
                src={require("assets/img/doc-page.jpeg")}
                to='/documents'
                width={`${width}px`} height={`${width /3}px`}
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
          </Grid>
            ) : null}
            
            <Grid item xs={12} md={4}>
              
                <img
                  alt="..."
                  className="img-raised"
                  src={require("assets/img/prof-page.jpeg")}
                  to='/profile-page'
                  width={`${width}px`} height={`${width/3}px`}
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
            </Grid>
            {largeScreen ? (
            <Grid item xs={12} md={4}>
              
                <img
                  alt="..."
                  className="img-raised"
                  src={require("assets/img/gallery-page.jpeg")}
                  to='/gallery'
                  width={`${width}px`} height={`${width /3}px`}
                ></img>
                
              <Button
                className="btn-round"
                color="default"
                to="/gallery"
                outline
                tag={Link}
              >
                View Gallery
              </Button>
            </Grid>
             ) : null}
             
             <Grid item xs={12} md={4}>
              
                <img
                  alt="..."
                  className="img-raised"
                  src={require("assets/img/trades-page.jpeg")}  // 这是一个假设的图片路径
                  to='/trades'
                  width={`${width}px`} height={`${width /3}px`}
                ></img>
                
              <Button
                className="btn-round"
                color="default"
                to="/trades"
                outline
                tag={Link}
              >
                View Trades
              </Button>
            </Grid>
          </Grid>
        </Container>
      </div>
    </>
  );
}

export default Examples;