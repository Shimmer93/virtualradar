using UnityEngine;
using RosSharp.RosBridgeClient;
using System.Collections.Generic;
using System.IO;

public class AnimatedHumanProcessing : MonoBehaviour
{
    // Animator related variables
    public Animator animator;
    public string animationName;
    private int numCurrSubframes = 0; // The current frame time

    // Radar related variables
    public GameObject radar;
    private ScreenSpaceRadarControlPlot ssrcp;
    private float subframeRate; // The frame rate of the radar
    private string keypointDir;
    private int idxTx;

    // Bone related variables
    private List<Transform> boneTransforms;

    void Awake()
    {
        ssrcp = radar.GetComponent<ScreenSpaceRadarControlPlot>();
        subframeRate = ssrcp.GetSubframeRate();
        keypointDir = ssrcp.GetDataDir() + "/keypoints";
        idxTx = ssrcp.GetIdxTx();

        Directory.CreateDirectory(keypointDir);
    }

    void Start()
    {
        animator.speed = 0; // Stop automatic playback

        if (idxTx != 0)
        {
            return;
        }
        // Initialize the list of bone transforms
        boneTransforms = new List<Transform>();

        // Get all the bone transforms from the animator
        using (StreamWriter writer = new StreamWriter(keypointDir + "/keypoint_names.csv", true))
        {
            foreach (HumanBodyBones bone in System.Enum.GetValues(typeof(HumanBodyBones)))
            {
                Transform boneTransform = animator.GetBoneTransform(bone);
                if (boneTransform != null)
                {
                    boneTransforms.Add(boneTransform);
                    writer.WriteLine($"{bone},{boneTransform.name}");
                }
            }
        }
    }

    void Update()
    {
        numCurrSubframes++;
        animator.Play(animationName, 0, numCurrSubframes / subframeRate); // Set the animation to the current frame
        animator.Update(1 / subframeRate); // Update the animation

        if (idxTx != 0)
        {
            return;
        }

        using (StreamWriter writer = new StreamWriter(keypointDir + "/keypoints.csv", true))
        {
            // Record the positions of the bones each frame
            foreach (Transform boneTransform in boneTransforms)
            {
                Vector3 position = boneTransform.position;
                writer.Write($"{position.x},{position.y},{position.z},");
            }

            writer.WriteLine(); // Separate frames with a blank line
        }

    }
}