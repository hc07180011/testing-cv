import axios from "axios";

const baseURL = process.env.REACT_APP_BASE_URL || "http://140.112.29.204:9696"
const instance = axios.create({
    baseURL: baseURL,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Content-Type": "multipart/form-data",
    },
})

const uploadVideo = (file, setLoading, setSimilarity, setSimilarityWide, setSuspects, setHorizontal, setVertical) => {

  setLoading(true)

  let formData = new FormData()
  formData.append("video", file)

  instance.post(
    "/flicker/create",
    formData,
  )
  .then((res) => {
    setSimilarity(res.data.similarities2)
    setSimilarityWide(res.data.similarities6)
    setSuspects(res.data.suspects)
    setHorizontal(res.data.horizontal_displacements)
    setVertical(res.data.vertical_displacements)
    setLoading(false)
  })
  .catch((err) => {
    console.log(err)
    setLoading(false)
  })
}

export {
    uploadVideo
}
