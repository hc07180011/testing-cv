import "./App.css";
import logo from "./logo.svg";
import { useState } from 'react';

import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Button from "@material-ui/core/Button";
import ClipLoader from "react-spinners/ClipLoader";
import { makeStyles } from "@material-ui/core/styles";

import { uploadVideo } from "./components/axios";
import { Figure, MixedFigure } from "./components/figure";

const useStyles = makeStyles((theme) => ({
  root: {
    margin: theme.spacing(1),
  },
  button: {
    margin: theme.spacing(1),
  },
  figure: {
    margin: theme.spacing(1),
  },
}))

const App = () => {
  
  const classes = useStyles()
  const [videoFile, setVideoFile] = useState(undefined)
  const [isLoading, setLoading] = useState(false)
  const [similarity, setSimilarity] = useState([])
  const [similarityWide, setSimilarityWide] = useState([])
  const [suspects, setSuspects] = useState([])
  const [horizontal, setHorizontal] = useState([])
  const [vertical, setVertical] = useState([])

  const handleFile = (files) => {
    if (files[0].size > 64 * 1024 * 1024) {
      alert("The uploading size limit is 64MB")
      return
    }
    setVideoFile(files[0])
    uploadVideo(files[0], setLoading, setSimilarity, setSimilarityWide, setSuspects, setHorizontal, setVertical)
  }

  return (
    <div className="App">
      <AppBar style={{ background: "rgb(216,234,245)", marginBottom: "1%" }} position="static">
        <Toolbar>
          <a href="/">
            <img
              alt="logo-img-main"
              src={logo}
              width="50px"
            />
          </a>
          Demo
        </Toolbar>
      </AppBar>
      <h2>Upload a video</h2>
      <Button
        component="label"
        variant="contained"
        color="primary"
        disabled={isLoading}  
        className={classes.button}
      >
        Upload
        <input
          type="file"
          hidden
          accept="video/*"
          onChange={
            (e) => handleFile(e.target.files)
          }
        />
      </Button>
      { videoFile ?
        <div>
          <h3>Preview</h3>
          <video width="30%" controls>
              <source src={URL.createObjectURL(videoFile)}/>
          </video>
        </div> :
        <div></div>
      }
      { isLoading ?
        <div className={classes.figure}><ClipLoader size={50} /></div> :
        <div className={classes.figure} style={{ display: similarity.length && !isLoading ? "inline" : "none" }}>
          <h2>Results:</h2>
          <MixedFigure
            data={[
              {
                index: [],
                value: similarity,
                type: "lines",
                name: "Similarity",
              },
              {
                index: [],
                value: similarityWide,
                type: "lines",
                name: "Similarity Wide",
              },
              {
                index: suspects,
                value: horizontal,
                type: "markers",
                name: "Horizontal Displacement",
              },
              {
                index: suspects,
                value: vertical,
                type: "markers",
                name: "Vertical Displacement",
              }
            ]}
          />
          <h3>Details:</h3>
          <Figure index={[]} mode={"lines"} value={similarity} title={"Similarity"}/>
          <Figure index={[]} mode={"lines"} value={similarityWide} title={"Similarity Wide"}/>
          <Figure index={suspects} mode={"markers"} value={horizontal} title={"Horizontal Displacement"}/>
          <Figure index={suspects} mode={"markers"} value={vertical} title={"Vertical Displacement"}/>
        </div>
      }
    </div>
  )
}

export default App;
