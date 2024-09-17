using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class KeypointRecorder : MonoBehaviour
{
    public Animator animator;
    public string saveFilePath = "Assets/keypoints.txt";

    private List<Transform> boneTransforms;

    void Start()
    {
        // Initialize the list of bone transforms
        boneTransforms = new List<Transform>();

        // Get all the bone transforms from the animator
        using (StreamWriter writer = new StreamWriter(saveFilePath, true))
        {
            foreach (HumanBodyBones bone in System.Enum.GetValues(typeof(HumanBodyBones)))
            {
                Transform boneTransform = animator.GetBoneTransform(bone);
                if (boneTransform != null)
                {
                    boneTransforms.Add(boneTransform);
                    writer.WriteLine($"{bone}, {boneTransform.name}");
                }
            }
            writer.WriteLine(); // Separate bone names with a blank line
        }
    }

    void Update()
    {
        using (StreamWriter writer = new StreamWriter(saveFilePath, true))
        {
            UnityEngine.Debug.Log(boneTransforms.Count);
            // Record the positions of the bones each frame
            foreach (Transform boneTransform in boneTransforms)
            {
                Vector3 position = boneTransform.position;
                writer.WriteLine($"{position.x}, {position.y}, {position.z}");
            }

            writer.WriteLine(); // Separate frames with a blank line
        }

    }
}