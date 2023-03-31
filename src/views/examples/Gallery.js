import React from "react";

import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import ImageListItemBar from '@mui/material/ImageListItemBar';
import DigitalArts from "views/documents/info/digital_arts";
import GalleryBg from "views/documents/info/gallery_bg";
import "./Gallery/ImageHover.css";

// reactstrap components
import {
  Button,
  // NavItem,
  // NavLink,
  // Nav,
  // TabContent,
  // TabPane,
  Container,
  // Row,
  // Col,
  UncontrolledTooltip
} from "reactstrap";

// core components
// import ExamplesNavbar from "components/Navbars/ExamplesNavbar.js";
import IndexNavbar from "components/Navbars/IndexNavbar";

import DefaultFooter from "components/Footers/DefaultFooter.js";
import GalleryHeader from "components/Headers/GalleryHeader";


function ImageListGallery({ itemData }) {
    return (
      <ImageList>
        {itemData.map((item) => (
          <ImageListItem key={item.img} >
            {item.type === "da" ? (
                <>
                <img
                    className="image-hover"
                    src={`${item.img}?w=248&fit=crop&auto=format`}
                    srcSet={`${item.img}?w=248&fit=crop&auto=format&dpr=2 2x`}
                    alt={item.title}
                    // style={{ width: '450px', height: '450px' }}
                    style={{ objectFit: "scale-down", width: "100%", height: "400px" }}
                    loading="lazy"
                    />
                  </> 
            ) : (
                <img
                    src={`${item.img}?w=248&fit=crop&auto=format`}
                    srcSet={`${item.img}?w=248&fit=crop&auto=format&dpr=2 2x`}
                    alt={item.title}
                    loading="lazy"
                    />
            )}
            
            <ImageListItemBar
              title={item.title}
              subtitle={item.author ? <span>by: {item.author}</span> : null}
              position="below"
            />
          </ImageListItem>
        ))}
      </ImageList>
    );
  }

function Gallery() {

  // const [pills, setPills] = React.useState("2");
  React.useEffect(() => {
    document.body.classList.add("profile-page");
    document.body.classList.add("sidebar-collapse");
    document.documentElement.classList.remove("nav-open");
    window.scrollTo(0, 0);
    document.body.scrollTop = 0;
    return function cleanup() {
      document.body.classList.remove("profile-page");
      document.body.classList.remove("sidebar-collapse");
    };
  }, []);

  return (
    <>
      <IndexNavbar label="Profile"/>
      <div className="wrapper">
        <GalleryHeader />
        <div className="section">
          <Container>
            <div className="button-container">
              
              <Button
                className="btn-round btn-icon"
                href="https://www.instagram.com/yuhengfan"
                color="danger"
                id="tooltipins"
                size="lg"
              >
                <i className="fab fa-instagram"></i>
              </Button>
              <UncontrolledTooltip delay={0} target="tooltipins">
                Follow me on Instagram
              </UncontrolledTooltip>
              <Button
                className="btn-round btn-icon"
                href="https://www.facebook.com/yuheng.fan.39/"
                color="info"
                id="facebook"
                size="lg"
              >
                <i className="fab fa-facebook" aria-hidden="true"></i>
              </Button>
              <UncontrolledTooltip delay={0} target="facebook">
                Follow me on Facebook
              </UncontrolledTooltip>

              <Button
                className="btn-round btn-icon"
                // href="https://www.linkedin.com/in/yuheng-fan-b915917b/"
                color="success"
                id="tooltipweixin"
                size="lg"
              >
                <i className="fab fa-weixin"></i>
              </Button>
              <UncontrolledTooltip delay={0} target="tooltipweixin">
                <img src={require("assets/img/qrcode.jpeg")} alt="QR code" style={{ width: '200px', height: '200px' }} />
              </UncontrolledTooltip>
            </div>
            <h2 className="title">Digital Arts</h2>

            <ImageListGallery itemData={DigitalArts}/>

            <h2 className="title">Backgrounds</h2>

            <ImageListGallery itemData={GalleryBg}/>
            
          </Container>
        </div>
        <DefaultFooter />
      </div>
    </>
  );
}

export default Gallery;
