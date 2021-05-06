// Create a folder (right click in the Assets directory, click Create>New Folder)
// and name it “Editor” if one doesn’t exist already. Place this script in that folder.

// This script creates a new menu item 'Synthesis3D > FBX to Prefab' in the main menu.
// Use it to create Prefab(s) from the selected GameObject(s).
// It will add RigidBody and Mesh Collider components to the mesh.
// It will be placed in the "Assets/Synthesis3D/Resources/Prefabs" folder.

// Ref: https://docs.unity3d.com/ScriptReference/PrefabUtility.html

using System.IO;
using UnityEngine;
using UnityEditor;

public class FBX2Prefab
{
    private static int LODs = 3;

    // Creates a new menu item 'Synthesis3D > FBX to Prefab' in the main menu.
    [MenuItem("Synthesis3D/FBX to Prefab")]
    static void CreatePrefab()
    {
        // Keep track of the currently selected GameObject(s)
        GameObject[] objectArray = Selection.gameObjects;

        // Loop through every GameObject in the array above
        foreach (GameObject gameObject in objectArray)
        {
            // modify LOD group percentage
            // Ref: https://forum.unity.com/threads/set-percentage-values-of-lodgroup.286206/
            LODGroup lodGroup = gameObject.GetComponent<LODGroup>();
            if (lodGroup != null) {            
                SerializedObject obj = new SerializedObject(lodGroup);
                SerializedProperty valArrProp = obj.FindProperty("m_LODs.Array");
                for (int i = 0; i < valArrProp.arraySize; i++) {
                    SerializedProperty sHeight = obj.FindProperty("m_LODs.Array.data[" + i.ToString() + "].screenRelativeHeight");
                    if (i == 0)
                        sHeight.doubleValue = 0.6;
                    if (i == 1)
                        sHeight.doubleValue = 0.3;
                    if (i == 2)
                        sHeight.doubleValue = 0.07;
                }
                obj.ApplyModifiedProperties();
            }

            // Add RigidBody Component (if it doesn't have one)
            if (gameObject.GetComponent<Rigidbody>() == null)
                gameObject.AddComponent<Rigidbody>();

            // Add Mesh Collider Component (if it doesn't have one)
            if (gameObject.GetComponent<MeshCollider>() == null)
            {
                MeshCollider mc = gameObject.AddComponent<MeshCollider>();
                mc.convex = true;
                MeshFilter childMesh = gameObject.transform.GetChild(LODs-1).GetComponent<MeshFilter>();
                mc.sharedMesh = childMesh.sharedMesh;// mesh for collide detection, use LOD2
            }
            
            // Set the path as within the Assets folder,
            // and name it as the GameObject's name with the .Prefab format
            if (!Directory.Exists("Assets/Synthesis3D/Resources/Prefabs/"))
                Directory.CreateDirectory("Assets/Synthesis3D/Resources/Prefabs/");
            string localPath = "Assets/Synthesis3D/Resources/Prefabs/" + gameObject.name + ".prefab";

            // Make sure the file name is unique, in case an existing Prefab has the same name.
            // localPath = AssetDatabase.GenerateUniqueAssetPath(localPath);
            // here we want to overwrite, so comment out this line

            // Create the new Prefab.
            PrefabUtility.SaveAsPrefabAssetAndConnect(gameObject, localPath, InteractionMode.UserAction);
        }
    }

    // Disable the menu item if no selection is in place.
    [MenuItem("Synthesis3D/FBX to Prefab", true)]
    static bool ValidateCreatePrefab()
    {
        return Selection.activeGameObject != null && !EditorUtility.IsPersistent(Selection.activeGameObject);
    }
}