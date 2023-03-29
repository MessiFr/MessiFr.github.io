import React from "react";

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
import ProfilePageHeader from "components/Headers/ProfilePageHeader.js";
import DefaultFooter from "components/Footers/DefaultFooter.js";

function ProfilePage() {
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
        <ProfilePageHeader />
        <div className="section">
          <Container>
            <div className="button-container">
              {/* <Button className="btn-round" color="info" size="lg">
                Follow
              </Button> */}
              <Button
                className="btn-round btn-icon"
                href="https://github.com/MessiFr"
                color="black"
                id="tooltip515203352"
                size="lg"
              >
                <i className="fab fa-github"></i>
              </Button>
              <UncontrolledTooltip delay={0} target="tooltip515203352">
                Follow me on Github
              </UncontrolledTooltip>
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
                href="https://www.linkedin.com/in/yuheng-fan-b915917b/"
                color="info"
                id="tooltip340339231"
                size="lg"
              >
                <i className="fab fa-linkedin-in"></i>
              </Button>
              <UncontrolledTooltip delay={0} target="tooltip340339231">
                Follow me on LinkedIn
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
            <h3 className="title">About me</h3>
            <h6 className="description">
              A student in The University of Melbourne, majoring Data Science.
            </h6>
            <h3 className='title'>Contacts</h3>
            <h5 className="category">Uni Email: yuhengf@student.unimelb.edu.au</h5>
            <h5 className="category">Personal Email: 15757470420@163.com</h5>
            <h5 className="category">Phone Number(+86): 15757470420</h5>
            <h5 className="category">Phone Number(+61): 0478598250</h5>
            
          </Container>
        </div>
        <DefaultFooter />
      </div>
    </>
  );
}

export default ProfilePage;
