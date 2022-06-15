import { useState } from 'react'
import Image from 'next/image'
import '../../styles/ImageUpload.module.css'

const endPoint = 'http://localhost:5000/predict'

export default function ImageUpload() {
  const [image, setImage] = useState(null)
  const [createObjectURL, setCreateObjectURL] = useState("/")
  const [prediction, setPrediction] = useState({})

  const uploadToClient = (event) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0]
      setImage(file)
      setCreateObjectURL(URL.createObjectURL(file))
    }
  }

  const uploadToServer = async (_event) => {
    const body = new FormData()
    body.append('file', image)
    fetch(endPoint, {
      method: 'POST',
      body
    }).then(response => response.json())
      .then(data => {
        setPrediction(data)
    })
      .catch(err => console.error(err))
  }

  return (
      <div>
        <Image
          src={createObjectURL}
          width={242}
          height={242}
          alt="image upload"
        />
        <h4>Select Image</h4>
        <input type="file" name="birdImage" onChange={uploadToClient} />
        <button className="btn btn-primary" onClick={uploadToServer}>
          Submit
        </button>
        <div>
          {JSON.stringify(prediction)}
        </div>
      </div>
  )
}

