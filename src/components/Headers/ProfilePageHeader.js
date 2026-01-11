import React from "react";

// reactstrap components
import { Container } from "reactstrap";
import { useHistory } from "react-router-dom";


import doc_counts from "views/documents/info/count";

// core components

function ProfilePageHeader() {
  let pageHeader = React.createRef();

  React.useEffect(() => {
    if (window.innerWidth > 991) {
      const updateScroll = () => {
        let windowScrollTop = window.pageYOffset / 3;
        if (pageHeader.current) { // 检查元素是否存在
          pageHeader.current.style.transform =
            "translate3d(0," + windowScrollTop + "px,0)";
        }
      };
      window.addEventListener("scroll", updateScroll);
      return function cleanup() {
        window.removeEventListener("scroll", updateScroll);
      };
    }
  });

  const history = useHistory();

  const handleClick = (pathName) => (e) => {
    e.preventDefault();
    history.push(`${pathName}`);
  };
  
  return (
    <>
      <div
        className="page-header clear-filter page-header-small"
        filter-color="blue"
      >
        <div
          className="page-header-image"
          style={{
            backgroundImage: "url(" + require("assets/img/profile_background.jpeg") + ")"
          }}
          ref={pageHeader}
        ></div>
        <Container>
          <div className="photo-container">
            <img alt="..." src={require("assets/img/avatar.png")}></img>
          </div>
          <h3 className="title">Yuheng Fan</h3>
          <p className="category">Data Scientist / Algorithm Engineer</p>
          <div className="content">
            <div className="social-description">
              {/* <a href="/documents" style={{ color:'white' }}> */}
                <h2 onClick={handleClick('/documents')}>{ doc_counts['documents'] }</h2>
              {/* </a> */}
              <p>Documents</p>
            </div>
            <div className="social-description">

              {/* <a href="/documents" style={{ color:'white' }}> */}
                <h2 onClick={handleClick('/documents')}>{ doc_counts['projects'] }</h2>
              {/* </a> */}
              <p>Projects</p>
            </div>
            <div className="social-description">
              <h2 onClick={ handleClick('/gallery') }>{ doc_counts['digit_arts'] }</h2>
              <p>Gallary</p>
            </div>
          </div>
        </Container>
      </div>
    </>
  );
}

export default ProfilePageHeader;