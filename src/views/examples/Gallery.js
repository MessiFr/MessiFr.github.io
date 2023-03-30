import React from "react";

import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import ImageListItemBar from '@mui/material/ImageListItemBar';

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
          <ImageListItem key={item.img}>
            {item.type === "da" ? (
                <img
                    src={`${item.img}?w=248&fit=crop&auto=format`}
                    srcSet={`${item.img}?w=248&fit=crop&auto=format&dpr=2 2x`}
                    alt={item.title}
                    style={{ width: '450px', height: '450px' }}
                    loading="lazy"
                    />
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

  const itemData = [
    {
      img: require("assets/img/digit_arts/ayanamirei.png"),
      title: 'Rei Ayanamirei',
      author: '@MessiFr',
      type: 'da'
    },
    {
      img: require("assets/img/digit_arts/ayanamirei2.png"),
      title: 'Rei Ayanamirei 2',
      author: '@MessiFr',
      type: 'da'
    },
    {
      img: require("assets/img/digit_arts/asuka.png"),
      title: 'Asuka',
      author: '@MessiFr',
      type: 'da'
    },
    {
      img: require("assets/img/digit_arts/asuka2.png"),
      title: 'Asuka 2',
      author: '@MessiFr',
      type: 'da'
    },
    {
      img: require("assets/img/digit_arts/asuka3.png"),
      title: 'Asuka 3',
      author: '@MessiFr',
      type: 'da'
    },
    {
      img: require("assets/img/digit_arts/Damonalbarn.png"),
      title: 'Damon Albarn',
      author: '@MessiFr',
      type: 'da'
    }
  ];

  const backgroundData = [
    {
        img: require("assets/img/bg_collection/room.jpg"),
        title: 'Room Illustration',
        author: 'Just Jon'
    },
    {
        img: require("assets/img/bg_collection/imaging.jpg"),
        title: 'Imagine',
    },
    {
        img: require("assets/img/bg_collection/reddeaddeption.jpg"),
        title: 'Red Dead Redemption',
    },
    {
        img: require("assets/img/bg_collection/island.jpg"),
        title: 'Island',
    },
    {
        img: require("assets/img/bg_collection/sakura.jpg"),
        title: 'Sakura',
    },
    {
        img: require("assets/img/bg_collection/leagueoflegends.jpg"),
        title: 'League of Legends',
    },
    {
        img: require("assets/img/bg_collection/planet.jpg"),
        title: 'Planet',
    },
    {
        img: require("assets/img/bg_collection/firewatchmodday.jpg"),
        title: 'Firewatch Mod Day',
    },
    {
        img: require("assets/img/bg_collection/firewatchmodnight.jpg"),
        title: 'Firewatch Mod Night',
    },
    {
        img: require("assets/img/bg_collection/fwmv2.jpg"),
        title: 'Firewatch Mod v2',
    },
    {
        img: require("assets/img/bg_collection/新璃月空港昼.jpg"),
        title: '新璃月空港 昼',
        author: '吃咖喱的poi'
    }
  ];

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

            <ImageListGallery itemData={itemData}/>

            <h2 className="title">Backgrounds</h2>

            <ImageListGallery itemData={backgroundData}/>
            
          </Container>
        </div>
        <DefaultFooter />
      </div>
    </>
  );
}

export default Gallery;
