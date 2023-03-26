// import { Container } from "@mui/material";
import React from "react";
import keplerGlReducer from 'kepler.gl/reducers';
import { createStore, combineReducers, applyMiddleware } from 'redux';
import { taskMiddleware } from 'react-palm/tasks';
import { Provider, useDispatch } from 'react-redux';
import KeplerGl from "kepler.gl";
import { addDataToMap } from "kepler.gl/actions";
import useSwr from "swr";
import mapConfig2 from '../../utils/mapConfig2'
import SERVER from '../../sections/@dashboard/app/config'; 

const reducers = combineReducers({
  keplerGl: keplerGlReducer,
});

const store = createStore(reducers, {}, applyMiddleware(taskMiddleware));

export default function Scenario2() {
  return (
      <Provider store={ store }>
        <ScenarioMap />
      </Provider>
  );
};

function ScenarioMap() {
  const dispatch = useDispatch();
  const { data } = useSwr("melbourne_twitter", async () => {
    const response = await fetch(
        `${SERVER}/api/map/scenario2`
    );
    
    const data = await response.json();
    console.log(data)
    return data;
  })

  mapConfig2.config.visState.layers[0].config.dataId = 'melbourne_twitter'

  React.useEffect(() => {
    if (data) {
      dispatch(addDataToMap({
        datasets: {
          info: {
            label: 'Melbourne Tweets Dataset',
            id: 'melbourne_twitter'
          },
          data: data
        },
        option: {
          centerMap: true,
          readOnly: false
        },
        config: mapConfig2
    }));
  }}, [dispatch, data]);

  return (
    <KeplerGl 
      id="melbourne_twitter" 
      mapboxApiAccessToken="pk.eyJ1IjoibWVzc2lmciIsImEiOiJjbDF5ZTNvbTMwYncxM2puMzZxbXJudHN5In0.Crs6xw83SHdNedU4DKK5fQ"
      width={window.innerWidth}
      height={window.innerHeight} 
    />
    // <div></div>
  );
};

